from decimal import Decimal

import pytest

from agentledger_treasury.core.degrader import BudgetVelocityTracker, DegradationEngine
from agentledger_treasury.core.treasury import Treasury


@pytest.mark.asyncio
async def test_no_degradation_when_budget_healthy():
    treasury = Treasury("s1", "1.00")
    velocity = BudgetVelocityTracker(expected_total_steps=4)
    degrader = DegradationEngine(treasury, velocity, critical_remaining_threshold=Decimal("0.20"))

    await treasury.request_allocation("agent-a", "0.10")
    resolved = degrader.resolve_model("agent-a", "claude-3-5-sonnet")
    assert resolved == "claude-3-5-sonnet"


@pytest.mark.asyncio
async def test_degrades_when_critical_and_burn_outstrips_progress():
    treasury = Treasury("s1", "1.00")
    velocity = BudgetVelocityTracker(expected_total_steps=10)
    degrader = DegradationEngine(treasury, velocity, critical_remaining_threshold=Decimal("0.90"))

    # Commit 85% of the budget while reporting 0 completed steps out of 10
    # expected -> utilization (0.85) far exceeds expected progress (0.0),
    # and remaining (0.15) is below the 0.90 threshold.
    await treasury.request_allocation("agent-a", "0.85")

    resolved = degrader.resolve_model("agent-a", "claude-3-5-sonnet")
    assert resolved == "claude-3-5-haiku"
    assert len(degrader.events) == 1
    assert degrader.events[0].original_model == "claude-3-5-sonnet"
    assert degrader.events[0].degraded_model == "claude-3-5-haiku"


@pytest.mark.asyncio
async def test_no_degradation_when_on_schedule_despite_low_budget():
    treasury = Treasury("s1", "1.00")
    velocity = BudgetVelocityTracker(expected_total_steps=10)
    degrader = DegradationEngine(treasury, velocity, critical_remaining_threshold=Decimal("0.90"))

    # Commit 85% of budget, but also report 9/10 steps done -> progress
    # (0.9) exceeds utilization (0.85), so burn is NOT outstripping
    # progress and no degradation should fire even though remaining < 90%.
    await treasury.request_allocation("agent-a", "0.85")
    for _ in range(9):
        velocity.record_step_completed()

    resolved = degrader.resolve_model("agent-a", "claude-3-5-sonnet")
    assert resolved == "claude-3-5-sonnet"
    assert len(degrader.events) == 0


@pytest.mark.asyncio
async def test_already_cheapest_tier_is_not_further_degraded():
    treasury = Treasury("s1", "1.00")
    velocity = BudgetVelocityTracker(expected_total_steps=10)
    degrader = DegradationEngine(treasury, velocity, critical_remaining_threshold=Decimal("0.90"))
    await treasury.request_allocation("agent-a", "0.85")

    resolved = degrader.resolve_model("agent-a", "claude-3-5-haiku")
    assert resolved == "claude-3-5-haiku"


def test_rewrite_runtime_config_does_not_mutate_input():
    treasury = Treasury("s1", "1.00")
    velocity = BudgetVelocityTracker(expected_total_steps=10)
    degrader = DegradationEngine(treasury, velocity, critical_remaining_threshold=Decimal("0.0"))
    original = {"model": "claude-3-5-sonnet", "temperature": 0.7}
    rewritten = degrader.rewrite_runtime_config("agent-a", original)
    assert original["model"] == "claude-3-5-sonnet"
    assert rewritten["temperature"] == 0.7
