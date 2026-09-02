from decimal import Decimal

import pytest

from agentledger.core.treasury import Treasury
from agentledger.exceptions import (
    AllocationAlreadySettledError,
    InsufficientTreasuryFundsError,
    SessionBudgetExhaustedError,
    UnknownAllocationTokenError,
)


@pytest.mark.asyncio
async def test_allocation_and_release_round_trip():
    treasury = Treasury("s1", "1.00")
    token = await treasury.request_allocation("agent-a", "0.30")
    assert treasury.available_usd == Decimal("0.70")
    assert treasury.committed_usd == Decimal("0.30")

    yielded = await treasury.release_allocation(token, "0.10")
    assert yielded == Decimal("0.20")
    assert treasury.available_usd == Decimal("0.90")
    assert treasury.committed_usd == Decimal("0")
    assert treasury.settled_spent_usd == Decimal("0.10")


@pytest.mark.asyncio
async def test_insufficient_funds_raises():
    treasury = Treasury("s1", "0.10")
    with pytest.raises(InsufficientTreasuryFundsError):
        await treasury.request_allocation("agent-a", "0.50")


@pytest.mark.asyncio
async def test_exhausted_budget_raises():
    treasury = Treasury("s1", "0.10")
    await treasury.request_allocation("agent-a", "0.10")
    with pytest.raises((InsufficientTreasuryFundsError, SessionBudgetExhaustedError)):
        await treasury.request_allocation("agent-b", "0.01")


@pytest.mark.asyncio
async def test_double_release_raises():
    treasury = Treasury("s1", "1.00")
    token = await treasury.request_allocation("agent-a", "0.10")
    await treasury.release_allocation(token, "0.05")
    with pytest.raises(AllocationAlreadySettledError):
        await treasury.release_allocation(token, "0.05")


@pytest.mark.asyncio
async def test_unknown_token_raises():
    treasury = Treasury("s1", "1.00")
    with pytest.raises(UnknownAllocationTokenError):
        await treasury.release_allocation("does-not-exist", "0.01")


@pytest.mark.asyncio
async def test_grant_extension_raises_ceiling():
    treasury = Treasury("s1", "1.00")
    token = await treasury.request_allocation("agent-a", "0.10")
    updated = await treasury.grant_extension(token, "0.05")
    assert updated.ceiling_usd == Decimal("0.15")
    assert treasury.committed_usd == Decimal("0.15")


@pytest.mark.asyncio
async def test_overspend_is_clamped_to_ceiling_on_release():
    treasury = Treasury("s1", "1.00")
    token = await treasury.request_allocation("agent-a", "0.10")
    yielded = await treasury.release_allocation(token, "999.00")
    assert yielded == Decimal("0")
    assert treasury.settled_spent_usd == Decimal("0.10")


@pytest.mark.asyncio
async def test_ledger_history_records_every_event():
    treasury = Treasury("s1", "1.00")
    token = await treasury.request_allocation("agent-a", "0.10")
    await treasury.release_allocation(token, "0.05")
    history = treasury.ledger_history()
    event_types = [e.event_type.value for e in history]
    assert "allocation_granted" in event_types
    assert "allocation_released" in event_types


@pytest.mark.asyncio
async def test_concurrent_allocations_never_oversubscribe_the_pool():
    import asyncio

    treasury = Treasury("s1", "1.00")

    async def try_allocate(agent_id: str):
        try:
            return await treasury.request_allocation(agent_id, "0.30")
        except (InsufficientTreasuryFundsError, SessionBudgetExhaustedError):
            return None

    results = await asyncio.gather(*[try_allocate(f"agent-{i}") for i in range(10)])
    granted = [r for r in results if r is not None]
    # Only floor(1.00 / 0.30) = 3 allocations of $0.30 can be granted.
    assert len(granted) == 3
    assert treasury.committed_usd == Decimal("0.90")
    assert treasury.available_usd == Decimal("0.10")
