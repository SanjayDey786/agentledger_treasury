"""
AgentLedger Treasury
=====================
A financial clearinghouse and dynamic credit-allocation middleware for
stateful, hierarchical multi-agent workflows (LangGraph, LangChain,
CrewAI). Treats token budgets as a fluid corporate treasury capable of
real-time micro-lending, credit yielding, and ROI-driven model degradation.

Quick start
-----------
    from decimal import Decimal
    from agentledger_treasury import Treasury, Reallocator, DegradationEngine, BudgetVelocityTracker
    from agentledger_treasury.integrations import ledger_guard

    treasury = Treasury(session_id="sess-1", total_session_budget_usd="1.00")
    reallocator = Reallocator(treasury)
    velocity = BudgetVelocityTracker(expected_total_steps=10)
    degrader = DegradationEngine(treasury, velocity)

    @ledger_guard("worker-a", treasury, max_usd_ceiling="0.20",
                  reallocator=reallocator, degradation_engine=degrader)
    async def worker_a(state: dict) -> dict:
        ...
"""

from agentledger_treasury.core.treasury import AllocationToken, Treasury, TreasuryLedgerEntry, TreasurySnapshot
from agentledger_treasury.core.reallocator import ProgressHook, Reallocator
from agentledger_treasury.core.degrader import BudgetVelocityTracker, DegradationEngine, DegradationEvent
from agentledger_treasury.pricing import DEFAULT_PRICING_TABLE, ModelPrice, PricingTable
from agentledger_treasury.exceptions import (
    AgentLedgerError,
    AllocationAlreadySettledError,
    AllocationExceededError,
    CreditExtensionDeniedError,
    InsufficientTreasuryFundsError,
    SessionBudgetExhaustedError,
    TreasuryError,
    UnknownAllocationTokenError,
    UnknownModelPricingError,
)
__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Core
    "Treasury",
    "TreasurySnapshot",
    "AllocationToken",
    "TreasuryLedgerEntry",
    "Reallocator",
    "ProgressHook",
    "DegradationEngine",
    "DegradationEvent",
    "BudgetVelocityTracker",
    # Pricing
    "PricingTable",
    "ModelPrice",
    "DEFAULT_PRICING_TABLE",
    # Exceptions
    "AgentLedgerError",
    "TreasuryError",
    "InsufficientTreasuryFundsError",
    "UnknownAllocationTokenError",
    "AllocationAlreadySettledError",
    "AllocationExceededError",
    "CreditExtensionDeniedError",
    "SessionBudgetExhaustedError",
    "UnknownModelPricingError",
    "PostgresTreasury",
]


def __getattr__(name: str) -> object:
    # Lazy so `import agentledger_treasury` works without the optional
    # `postgres` extra (`asyncpg`) installed; only resolved on first access.
    if name == "PostgresTreasury":
        from agentledger_treasury.backends import PostgresTreasury

        return PostgresTreasury
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
