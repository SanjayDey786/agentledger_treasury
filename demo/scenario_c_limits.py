"""
Scenario C -- where reallocation does NOT help. The honest half of the demo.

C1: Systemic overrun. If every agent in the session is running hot at once,
    there is no idle slack anywhere in the pool to redistribute. Give-back
    only works because SOME agents finish under budget; if none do, the
    ledger policy converges toward the same failure rate as the naive
    policy -- it reallocates money, it does not create it.

C2: A solo agent with no siblings. Micro-lending can only fund a shortfall
    out of money OTHER agents already returned. With one agent and no
    siblings, there is nothing to borrow from, so a single unpredictable
    call can still sink the session even with every governance feature
    turned on. This also makes concrete a point from earlier discussion:
    by the time the treasury denies the extension, the real LLM call has
    already happened and (in a real deployment) already been billed by the
    provider -- the treasury's denial changes what your application does
    NEXT, it cannot claw back money already spent upstream.
"""

from __future__ import annotations

import asyncio
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fake_llm import call_llm  # noqa: E402
from pg import make_pool, reset_session  # noqa: E402
from agentledger_treasury.backends.postgres import PostgresTreasury  # noqa: E402
from agentledger_treasury.core.reallocator import Reallocator  # noqa: E402
from agentledger_treasury.exceptions import (  # noqa: E402
    CreditExtensionDeniedError,
    InsufficientTreasuryFundsError,
    SessionBudgetExhaustedError,
)
from agentledger_treasury.pricing import DEFAULT_PRICING_TABLE  # noqa: E402

MODEL = "claude-3-5-haiku"
PROMPT_TOKENS = 150
EXPECTED_OUTPUT_TOKENS = 300

_nominal_cost = DEFAULT_PRICING_TABLE.estimate_cost(MODEL, PROMPT_TOKENS, EXPECTED_OUTPUT_TOKENS)


async def run_ledger_trial(rng, treasury, reallocator, agents, per_agent_share, variance, runaway_prob) -> bool:
    for agent in agents:
        try:
            token = await treasury.request_allocation(agent, per_agent_share, model_name=MODEL)
        except (InsufficientTreasuryFundsError, SessionBudgetExhaustedError):
            return False

        result = call_llm(
            rng,
            model=MODEL,
            prompt_tokens=PROMPT_TOKENS,
            expected_output_tokens=EXPECTED_OUTPUT_TOKENS,
            variance=variance,
            runaway_probability=runaway_prob,
        )
        actual_cost = DEFAULT_PRICING_TABLE.estimate_cost(MODEL, result.prompt_tokens, result.completion_tokens)

        if actual_cost > token.ceiling_usd:
            shortfall = actual_cost - token.ceiling_usd
            try:
                await reallocator.request_credit_extension(agent, token, shortfall, force=True)
            except CreditExtensionDeniedError:
                await treasury.release_allocation(token, token.ceiling_usd)
                return False

        await treasury.release_allocation(token, actual_cost)
    return True


async def run_naive_trial(rng, agents, per_agent_share, variance, runaway_prob) -> bool:
    for agent in agents:
        result = call_llm(
            rng,
            model=MODEL,
            prompt_tokens=PROMPT_TOKENS,
            expected_output_tokens=EXPECTED_OUTPUT_TOKENS,
            variance=variance,
            runaway_probability=runaway_prob,
        )
        actual_cost = DEFAULT_PRICING_TABLE.estimate_cost(MODEL, result.prompt_tokens, result.completion_tokens)
        if actual_cost > per_agent_share:
            return False
    return True


async def demo_c1(pool, n_trials: int = 150) -> None:
    print("=" * 70)
    print("C1 -- Systemic overrun: every agent is running hot, nobody has slack")
    print("=" * 70)

    agents = ["researcher", "analyst", "synthesizer"]
    # Zero slack (share == nominal expected cost) and high variance/runaway
    # probability: on average, roughly half of all calls will exceed their
    # own share, so there is rarely an under-budget agent anywhere in the
    # session for the ledger to borrow from.
    per_agent_share = _nominal_cost
    variance = 0.55
    runaway_prob = 0.20

    naive_successes = 0
    ledger_successes = 0
    for trial in range(n_trials):
        seed = trial
        naive_ok = await run_naive_trial(random.Random(seed), agents, per_agent_share, variance, runaway_prob)

        session_id = f"scenario-c1-trial-{trial}"
        await reset_session(session_id)
        total_budget = per_agent_share * len(agents)
        treasury = PostgresTreasury(
            pool, session_id=session_id, total_session_budget_usd=str(total_budget), create_tables=True
        )
        reallocator = Reallocator(treasury)
        ledger_ok = await run_ledger_trial(
            random.Random(seed), treasury, reallocator, agents, per_agent_share, variance, runaway_prob
        )

        naive_successes += int(naive_ok)
        ledger_successes += int(ledger_ok)

    naive_pct = naive_successes / n_trials
    ledger_pct = ledger_successes / n_trials
    print(f"naive:  {naive_successes}/{n_trials} succeeded ({naive_pct:.0%})")
    print(f"ledger: {ledger_successes}/{n_trials} succeeded ({ledger_pct:.0%})")
    print(f"\n>>> The ledger still wins some of the time ({ledger_successes - naive_successes} more sessions) --")
    print("    even at zero nominal slack, variance occasionally leaves ONE agent a little under budget by")
    print("    luck, and that scrap is enough to save another. But compare the ABSOLUTE rescue count to")
    print("    Scenario B, where the same mechanism rescued far more sessions out of the same trial count:")
    print("    when demand saturates the pool, both policies collapse toward the same low success rate --")
    print(f"    reallocation redistributes slack, it does not create it, and when both are near {naive_pct:.0%}")
    print("    there just isn't much slack left anywhere to redistribute.")


async def demo_c2(pool) -> None:
    print("\n" + "=" * 70)
    print("C2 -- A solo agent has no siblings to borrow from")
    print("=" * 70)

    session_id = "scenario-c2-solo"
    await reset_session(session_id)
    # Budget sized for exactly one nominal-sized call -- no slack, no
    # siblings to have given anything back.
    total_budget = _nominal_cost
    treasury = PostgresTreasury(
        pool, session_id=session_id, total_session_budget_usd=str(total_budget), create_tables=True
    )
    reallocator = Reallocator(treasury)

    print(f"Total budget: ${total_budget}  (sized for exactly one expected-size call, one agent, no siblings)")

    token = await treasury.request_allocation("solo-agent", total_budget, model_name=MODEL)

    # Force a runaway completion: the model produced 8x more output than
    # expected -- the kind of thing no pre-flight token-count estimate can
    # see coming, and nothing capped it in real time because this library
    # (like effectively every cost-governance tool of this kind) does not
    # inject a hard `max_tokens` ceiling into the actual provider call.
    rng = random.Random(42)
    result = call_llm(
        rng,
        model=MODEL,
        prompt_tokens=PROMPT_TOKENS,
        expected_output_tokens=EXPECTED_OUTPUT_TOKENS,
        runaway_probability=1.0,
        runaway_multiplier=8.0,
    )
    actual_cost = DEFAULT_PRICING_TABLE.estimate_cost(MODEL, result.prompt_tokens, result.completion_tokens)
    print(f"Real completion came back at {result.completion_tokens} output tokens "
          f"(expected ~{EXPECTED_OUTPUT_TOKENS}) -> real cost ${actual_cost}, ceiling was ${token.ceiling_usd}")

    print("\nIn a REAL deployment, this call has ALREADY happened and the provider has ALREADY billed")
    print("for those tokens by the time this next line runs. Nothing below can undo that spend --")
    print("it can only decide what the ledger records and what your application does next.")

    if actual_cost > token.ceiling_usd:
        shortfall = actual_cost - token.ceiling_usd
        try:
            await reallocator.request_credit_extension("solo-agent", token, shortfall, force=True)
            print("(unexpected: extension was granted)")
        except CreditExtensionDeniedError as exc:
            await treasury.release_allocation(token, token.ceiling_usd)
            print(f"\n>>> CreditExtensionDeniedError: {exc}")
            print(">>> Correct behavior: there was no sibling agent's give-back anywhere in this pool")
            print("    to fund the shortfall, so the ledger honestly reports the session as over-budget")
            print("    instead of silently absorbing a cost it can't actually cover.")


async def main() -> None:
    pool = await make_pool()
    await demo_c1(pool)
    await demo_c2(pool)
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
