"""
agentledger_treasury.exceptions
=======================
Centralized exception hierarchy for AgentLedger. All errors raised by the
treasury, reallocator, degrader, and integration layers derive from
`AgentLedgerError` so callers can catch broadly or narrowly as needed.
"""

from __future__ import annotations

from decimal import Decimal


class AgentLedgerError(Exception):
    """Base class for all AgentLedger exceptions."""


class TreasuryError(AgentLedgerError):
    """Raised for generic treasury-level failures."""


class InsufficientTreasuryFundsError(TreasuryError):
    """Raised when the central Treasury Pool cannot fund a requested allocation."""

    def __init__(self, requested_usd: Decimal, available_usd: Decimal) -> None:
        self.requested_usd = requested_usd
        self.available_usd = available_usd
        super().__init__(
            f"Insufficient treasury funds: requested ${requested_usd:.6f}, "
            f"only ${available_usd:.6f} available in the central pool."
        )


class UnknownAllocationTokenError(TreasuryError):
    """Raised when an operation references an `AllocationToken` that does not exist."""

    def __init__(self, token_id: str) -> None:
        self.token_id = token_id
        super().__init__(f"Unknown or already-settled allocation token: {token_id!r}")


class AllocationAlreadySettledError(TreasuryError):
    """Raised when attempting to release or extend a token that has already been settled."""

    def __init__(self, token_id: str) -> None:
        self.token_id = token_id
        super().__init__(f"Allocation token {token_id!r} has already been settled.")


class AllocationExceededError(AgentLedgerError):
    """
    Raised (or attached to a mid-run interception) when an agent's actual spend
    would exceed its currently active AllocationToken ceiling and no credit
    extension could be secured.
    """

    def __init__(self, agent_id: str, token_id: str, attempted_usd: Decimal, ceiling_usd: Decimal) -> None:
        self.agent_id = agent_id
        self.token_id = token_id
        self.attempted_usd = attempted_usd
        self.ceiling_usd = ceiling_usd
        super().__init__(
            f"Agent {agent_id!r} exceeded allocation token {token_id!r}: "
            f"attempted ${attempted_usd:.6f} against ceiling ${ceiling_usd:.6f}."
        )


class CreditExtensionDeniedError(AgentLedgerError):
    """Raised when a micro-lending credit extension request is rejected by the Treasury."""

    def __init__(self, agent_id: str, requested_usd: Decimal, reason: str) -> None:
        self.agent_id = agent_id
        self.requested_usd = requested_usd
        self.reason = reason
        super().__init__(
            f"Credit extension of ${requested_usd:.6f} denied for agent {agent_id!r}: {reason}"
        )


class SessionBudgetExhaustedError(AgentLedgerError):
    """Raised when the entire session's Total_Session_Budget has been depleted."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session {session_id!r} has exhausted its total session budget.")


class UnknownModelPricingError(AgentLedgerError):
    """Raised when cost estimation is requested for a model absent from the pricing table."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        super().__init__(
            f"No pricing entry for model {model_name!r}. Register it via "
            f"`PricingTable.register(...)` before use."
        )
