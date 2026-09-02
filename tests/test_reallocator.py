from decimal import Decimal

import pytest

from agentledger.core.reallocator import Reallocator
from agentledger.core.treasury import Treasury
from agentledger.exceptions import CreditExtensionDeniedError


@pytest.mark.asyncio
async def test_give_back_yields_unused_delta():
    treasury = Treasury("s1", "1.00")
    reallocator = Reallocator(treasury)
    token = await treasury.request_allocation("agent-a", "0.20")
    yielded = await reallocator.give_back(token, "0.05")
    assert yielded == Decimal("0.15")
    assert treasury.available_usd == Decimal("0.95")


@pytest.mark.asyncio
async def test_extension_denied_below_progress_threshold():
    treasury = Treasury("s1", "1.00")
    reallocator = Reallocator(treasury, min_progress_for_extension=0.75)
    token = await treasury.request_allocation("agent-a", "0.10")
    reallocator.report_progress("agent-a", 0.2)
    with pytest.raises(CreditExtensionDeniedError):
        await reallocator.request_credit_extension("agent-a", token, "0.05")


@pytest.mark.asyncio
async def test_extension_granted_above_progress_threshold():
    treasury = Treasury("s1", "1.00")
    reallocator = Reallocator(treasury, min_progress_for_extension=0.75)
    token = await treasury.request_allocation("agent-a", "0.10")
    reallocator.report_progress("agent-a", 0.9)
    updated = await reallocator.request_credit_extension("agent-a", token, "0.05")
    assert updated.ceiling_usd == Decimal("0.15")


@pytest.mark.asyncio
async def test_extension_denied_when_pool_insufficient():
    treasury = Treasury("s1", "0.10")
    reallocator = Reallocator(treasury, min_progress_for_extension=0.0)
    token = await treasury.request_allocation("agent-a", "0.10")
    with pytest.raises(CreditExtensionDeniedError):
        await reallocator.request_credit_extension("agent-a", token, "0.05")


@pytest.mark.asyncio
async def test_yielded_savings_fund_a_sibling_extension():
    """Mirrors the STEP 4 rescue scenario at the unit level."""
    treasury = Treasury("s1", "0.30")
    reallocator = Reallocator(treasury, min_progress_for_extension=0.5)

    token_a = await treasury.request_allocation("agent-a", "0.20")
    await reallocator.give_back(token_a, "0.05")  # yields $0.15 back

    token_b = await treasury.request_allocation("agent-b", "0.05")
    reallocator.report_progress("agent-b", 0.8)
    # agent-b needs $0.08 more than its ceiling; only fundable because of
    # agent-a's yielded savings.
    updated = await reallocator.request_credit_extension("agent-b", token_b, "0.08")
    assert updated.ceiling_usd == Decimal("0.13")
