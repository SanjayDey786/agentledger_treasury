"""
AgentLedger
============
A financial clearinghouse and dynamic credit-allocation middleware for
stateful, hierarchical multi-agent workflows (LangGraph, LangChain,
CrewAI). Treats token budgets as a fluid corporate treasury capable of
real-time micro-lending, credit yielding, and ROI-driven model degradation.

Quick start
-----------
    from decimal import Decimal
    from agentledger import Treasury, Reallocator, DegradationEngine, BudgetVelocityTracker
    from agentledger.integrations import ledger_guard

    treasury = Treasury(session_id="sess-1", total_session_budget_usd="1.00")
    reallocator = Reallocator(treasury)
    velocity = BudgetVelocityTracker(expected_total_steps=10)
    degrader = DegradationEngine(treasury, velocity)

    @ledger_guard("worker-a", treasury, max_usd_ceiling="0.20",
                  reallocator=reallocator, degradation_engine=degrader)
    async def worker_a(state: dict) -> dict:
        ...
"""

from agentledger.core.treasury import AllocationToken, Treasury, TreasuryLedgerEntry, TreasurySnapshot
from agentledger.core.reallocator import ProgressHook, Reallocator
from agentledger.core.degrader import BudgetVelocityTracker, DegradationEngine, DegradationEvent
from agentledger.pricing import DEFAULT_PRICING_TABLE, ModelPrice, PricingTable
from agentledger.exceptions import (
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
from agentledger.backends import PostgresTreasury

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
