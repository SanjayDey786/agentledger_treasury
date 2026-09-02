"""
examples/supervisor_worker_example.py
========================================
STEP 4 of the PRD: a concrete integration example.

A Supervisor Agent spins up two worker agents against a shared session
Treasury:

  * `researcher` is allocated a modest ceiling, finishes comfortably under
    budget on a short task, and its unspent delta is immediately "given
    back" to the Treasury Pool (Module B).

  * `synthesizer` is allocated a tight ceiling but ends up needing more
    tokens than expected (a long synthesis pass). It reports task progress
    as it works; once it crosses the configured progress threshold and
    hits its local ceiling, it requests a micro-lending credit extension.
    Because `researcher` already yielded its savings back to the pool, the
    Treasury can fund the extension without breaching the user's total
    session cap -- rescuing `synthesizer` from what would otherwise be a
    budget crash.

This example uses a `FakeLLM` stand-in (no network calls, no API keys
required) so it runs standalone with `python examples/supervisor_worker_example.py`.
It illustrates the exact call shape (`.ainvoke` / usage dicts) that a real
LangChain `ChatAnthropic` or `ChatOpenAI` runnable would produce, so wiring
in a real model is a drop-in swap.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any, Dict

from agentledger import BudgetVelocityTracker, DegradationEngine, Reallocator, Treasury
from agentledger.exceptions import AllocationExceededError, CreditExtensionDeniedError
from agentledger.integrations.langgraph import ledger_guard


class FakeLLM:
    """
    Stands in for a real chat-model call. Returns a dict shaped like a
    LangGraph node's output, including a LiteLLM-style `usage` block so
    `agentledger.usage.extract_usage` can settle real cost against it.
    """

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    async def acall(self, model: str) -> Dict[str, Any]:
        # Simulate network latency for realism without slowing the example down.
        await asyncio.sleep(0.01)
        return {
            "text": f"[{model}] simulated completion",
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "model": model,
            },
        }


async def main() -> None:
    # ------------------------------------------------------------------ #
    # Session setup: a $0.02 total budget shared by both workers -- small
    # on purpose so the example clearly demonstrates the rescue scenario
    # without needing enormous simulated token counts.
    # ------------------------------------------------------------------ #
    treasury = Treasury(session_id="demo-session", total_session_budget_usd="0.020000")
    reallocator = Reallocator(treasury, min_progress_for_extension=0.5)
    velocity = BudgetVelocityTracker(expected_total_steps=2)
    degrader = DegradationEngine(treasury, velocity, critical_remaining_threshold=Decimal("0.20"))

    print(f"=== AgentLedger Demo: Session '{treasury.session_id}' ===")
    print(f"Total session budget: ${treasury.total_session_budget_usd}\n")

    # ------------------------------------------------------------------ #
    # Worker 1: researcher -- allocated $0.012, actually only needs ~$0.004.
    # ------------------------------------------------------------------ #
    researcher_llm = FakeLLM(prompt_tokens=300, completion_tokens=200)  # cheap, short task

    @ledger_guard(
        "researcher",
        treasury,
        max_usd_ceiling="0.012000",
        reallocator=reallocator,
        degradation_engine=degrader,
        prompt_key="input",
        model_key="model",
    )
    async def researcher_node(state: Dict[str, Any]) -> Dict[str, Any]:
        result = await researcher_llm.acall(state["model"])
        velocity.record_step_completed()
        return {**state, **result}

    researcher_state = {"input": "Summarize the latest paper on agentic budgeting.", "model": "claude-3-5-sonnet"}
    researcher_result = await researcher_node(researcher_state)

    snap = treasury.snapshot()
    print("-- researcher finished --")
    print(f"  output: {researcher_result['text']}")
    print(f"  treasury available after researcher: ${snap.available_usd:.6f}\n")

    # ------------------------------------------------------------------ #
    # Worker 2: synthesizer -- allocated a tight $0.004 ceiling but its
    # actual synthesis pass runs long (simulated by a much larger
    # completion token count), so it will need a rescue extension.
    # ------------------------------------------------------------------ #
    synthesizer_llm = FakeLLM(prompt_tokens=400, completion_tokens=600)  # expensive, long synthesis

    @ledger_guard(
        "synthesizer",
        treasury,
        max_usd_ceiling="0.004000",
        reallocator=reallocator,
        degradation_engine=degrader,
        prompt_key="input",
        model_key="model",
    )
    async def synthesizer_node(state: Dict[str, Any]) -> Dict[str, Any]:
        # Report meaningful progress before the expensive call so a
        # subsequent micro-lending request (triggered automatically by the
        # guard on post-flight overage) is honored by the Reallocator.
        reallocator.report_progress("synthesizer", 0.8)
        result = await synthesizer_llm.acall(state["model"])
        velocity.record_step_completed()
        return {**state, **result}

    synthesizer_state = {
        "input": "Synthesize the researcher's findings into a full report.",
        "model": "claude-3-5-sonnet",
    }

    try:
        synthesizer_result = await synthesizer_node(synthesizer_state)
    except (AllocationExceededError, CreditExtensionDeniedError) as exc:
        print("-- synthesizer CRASHED despite rescue attempt --")
        print(f"  {exc}\n")
        synthesizer_result = None

    if synthesizer_result is not None:
        print("-- synthesizer finished (rescued via micro-lending) --")
        print(f"  output: {synthesizer_result['text']}")

    final_snap = treasury.snapshot()
    print("\n=== Final Treasury State ===")
    print(f"  total_session_budget_usd : ${final_snap.total_session_budget_usd:.6f}")
    print(f"  settled_spent_usd        : ${final_snap.settled_spent_usd:.6f}")
    print(f"  available_usd            : ${final_snap.available_usd:.6f}")
    print(f"  active_token_count       : {final_snap.active_token_count}")
    print(f"  settled_token_count      : {final_snap.settled_token_count}")

    print("\n=== Full Ledger Audit Trail ===")
    for entry in treasury.ledger_history():
        print(
            f"  [{entry.event_type.value:>26}] agent={entry.agent_id:<12} "
            f"amount=${entry.amount_usd:.6f} available_after=${entry.treasury_available_after_usd:.6f} "
            f"{entry.metadata}"
        )

    if degrader.events:
        print("\n=== Degradation Events ===")
        for event in degrader.events:
            print(f"  {event.agent_id}: {event.original_model} -> {event.degraded_model} ({event.reason})")


if __name__ == "__main__":
    asyncio.run(main())
