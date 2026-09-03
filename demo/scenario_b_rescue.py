"""
Scenario B -- where give-back + micro-lending actually helps.

Same underlying "reality" (identical simulated per-agent token usage) is
replayed against two governance policies, many times, so the comparison is
apples-to-apples per trial rather than aggregate noise:

  - "naive": the total budget is split evenly across N agents up front.
    If any agent's real cost exceeds its fixed share, that agent (and the
    session) fails. No agent can use another agent's slack.

  - "ledger": the SAME total budget lives in one shared PostgresTreasury
    session_id (per trial). Agents request allocations sized to their
    nominal share; whichever finish under budget give back the unused
    delta immediately; whichever run over their ceiling request a
    micro-lending extension funded only by that returned slack. The
    session only fails if the shared pool genuinely can't cover the
    shortfall even after reallocation.

Both policies see the exact same random draws per trial (same seed), so
any difference in outcome is caused by the governance strategy, not by
lucky/unlucky randomness.
"""

from __future__ import annotations

import asyncio
import random
import sys
import os
from decimal import Decimal

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
AGENTS = ["researcher", "analyst", "synthesizer"]
PROMPT_TOKENS = 150
EXPECTED_OUTPUT_TOKENS = 300
VARIANCE = 0.35
RUNAWAY_PROBABILITY = 0.06  # ~6% of calls blow way past their expectation
N_TRIALS = 150

# Nominal per-agent budget with some slack over the "expected" cost --
# a realistic setting: you sized the budget off your typical/expected
# usage, not off worst-case, because worst-case would defeat the point of
# having a budget at all.
_nominal_cost = DEFAULT_PRICING_TABLE.estimate_cost(MODEL, PROMPT_TOKENS, EXPECTED_OUTPUT_TOKENS)
PER_AGENT_SHARE = (_nominal_cost * Decimal("1.15")).quantize(Decimal("0.000001"))
TOTAL_BUDGET = PER_AGENT_SHARE * len(AGENTS)


async def run_naive_trial(rng: random.Random) -> bool:
    """Returns True if the session succeeded (every agent stayed within its fixed share)."""
    for agent in AGENTS:
        result = call_llm(
            rng,
            model=MODEL,
            prompt_tokens=PROMPT_TOKENS,
            expected_output_tokens=EXPECTED_OUTPUT_TOKENS,
            variance=VARIANCE,
            runaway_probability=RUNAWAY_PROBABILITY,
        )
        actual_cost = DEFAULT_PRICING_TABLE.estimate_cost(
            MODEL, result.prompt_tokens, result.completion_tokens
        )
        if actual_cost > PER_AGENT_SHARE:
            return False
    return True


async def run_ledger_trial(rng: random.Random, treasury: PostgresTreasury, reallocator: Reallocator) -> bool:
    """Same random draws, but governed by the shared treasury with give-back + lending."""
    for agent in AGENTS:
        try:
            token = await treasury.request_allocation(agent, PER_AGENT_SHARE, model_name=MODEL)
        except (InsufficientTreasuryFundsError, SessionBudgetExhaustedError):
            return False

        result = call_llm(
            rng,
            model=MODEL,
            prompt_tokens=PROMPT_TOKENS,
            expected_output_tokens=EXPECTED_OUTPUT_TOKENS,
            variance=VARIANCE,
            runaway_probability=RUNAWAY_PROBABILITY,
        )
        actual_cost = DEFAULT_PRICING_TABLE.estimate_cost(
            MODEL, result.prompt_tokens, result.completion_tokens
        )

        if actual_cost > token.ceiling_usd:
            shortfall = actual_cost - token.ceiling_usd
            # `force=True`: this synthetic demo has no real progress hook,
            # so we treat "the call already finished and we know its real
            # cost" as equivalent to "fully progressed" -- exactly the
            # judgment call `LedgerGuardMiddleware._settle` makes in the
            # real integration.
            try:
                await reallocator.request_credit_extension(agent, token, shortfall, force=True)
            except CreditExtensionDeniedError:
                await treasury.release_allocation(token, token.ceiling_usd)
                return False

        await treasury.release_allocation(token, actual_cost)
    return True


async def main() -> None:
    print("=" * 70)
    print("Scenario B -- give-back + micro-lending vs. a fixed even split")
    print(f"Model: {MODEL}   Agents per session: {len(AGENTS)}   Trials: {N_TRIALS}")
    print(f"Per-agent nominal share: ${PER_AGENT_SHARE}   Total budget: ${TOTAL_BUDGET}")
    print(f"Output-token variance: log-normal(sigma={VARIANCE}), {RUNAWAY_PROBABILITY:.0%} chance of an 8x runaway call")
    print("=" * 70)

    pool = await make_pool()

    naive_successes = 0
    ledger_successes = 0

    for trial in range(N_TRIALS):
        seed = trial  # identical seed => identical simulated reality for both policies
        naive_ok = await run_naive_trial(random.Random(seed))

        session_id = f"scenario-b-trial-{trial}"
        await reset_session(session_id)
        treasury = PostgresTreasury(
            pool, session_id=session_id, total_session_budget_usd=str(TOTAL_BUDGET), create_tables=True
        )
        reallocator = Reallocator(treasury)
        ledger_ok = await run_ledger_trial(random.Random(seed), treasury, reallocator)

        naive_successes += int(naive_ok)
        ledger_successes += int(ledger_ok)

    await pool.close()

    print(f"\nnaive (fixed even split):   {naive_successes}/{N_TRIALS} sessions succeeded "
          f"({naive_successes / N_TRIALS:.0%})")
    print(f"ledger (give-back+lending): {ledger_successes}/{N_TRIALS} sessions succeeded "
          f"({ledger_successes / N_TRIALS:.0%})")
    print(f"\n>>> {ledger_successes - naive_successes} additional sessions rescued out of {N_TRIALS}, "
          f"purely by reallocating money that already existed in the pool -- no extra budget was spent.")


if __name__ == "__main__":
    asyncio.run(main())
