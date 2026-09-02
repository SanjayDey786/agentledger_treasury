"""
agentledger.core.treasury
==========================
Module A: The Treasury Clearinghouse.

Implements the central, thread-safe, atomic ledger for a single multi-agent
session. All USD amounts are represented as `decimal.Decimal` to avoid
floating point drift across potentially thousands of micro-transactions.

Concurrency model
------------------
Bookkeeping operations here (dict mutation + Decimal arithmetic) are pure,
in-memory, and non-blocking -- they never perform I/O and never `await`
anything while holding the lock. We therefore protect the ledger state with
a single `threading.RLock`, which is safe to acquire synchronously from
*both* plain multi-threaded callers (e.g. a CrewAI thread pool) and from
`async def` methods running on an asyncio event loop, since the critical
sections are always short and non-yielding. This gives us true thread-safety
without the pitfalls of mixing `asyncio.Lock` across threads, while still
exposing a fully `async/await`-native public API as required.
"""

from __future__ import annotations

import itertools
import threading
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

from agentledger.exceptions import (
    AllocationAlreadySettledError,
    InsufficientTreasuryFundsError,
    SessionBudgetExhaustedError,
    UnknownAllocationTokenError,
)


class TokenStatus(str, Enum):
    ACTIVE = "active"
    SETTLED = "settled"


class LedgerEventType(str, Enum):
    ALLOCATION_GRANTED = "allocation_granted"
    ALLOCATION_DENIED = "allocation_denied"
    ALLOCATION_RELEASED = "allocation_released"
    CREDIT_EXTENSION_GRANTED = "credit_extension_granted"
    CREDIT_EXTENSION_DENIED = "credit_extension_denied"
    MODEL_DEGRADED = "model_degraded"


@dataclass(frozen=True)
class TreasuryLedgerEntry:
    """An immutable audit-log entry recorded for every treasury operation."""

    event_id: str
    event_type: LedgerEventType
    timestamp: float
    agent_id: str
    token_id: Optional[str]
    amount_usd: Decimal
    treasury_available_after_usd: Decimal
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class AllocationToken:
    """
    Represents an agent's active claim on a slice of the Treasury Pool.

    `ceiling_usd` is mutable over the token's lifetime because credit
    extensions (Module B micro-lending) raise it. `initial_ceiling_usd`
    is preserved for reporting/audit purposes.
    """

    token_id: str
    agent_id: str
    initial_ceiling_usd: Decimal
    ceiling_usd: Decimal
    model_name: Optional[str]
    created_at: float
    status: TokenStatus = TokenStatus.ACTIVE
    actual_spent_usd: Decimal = Decimal("0")
    extensions_granted_usd: Decimal = Decimal("0")

    @property
    def remaining_usd(self) -> Decimal:
        """Headroom left on this token before it hits its own ceiling."""
        return self.ceiling_usd - self.actual_spent_usd

    def would_exceed(self, attempted_total_spend_usd: Decimal) -> bool:
        return attempted_total_spend_usd > self.ceiling_usd


class Treasury:
    """
    The central, per-session Treasury Clearinghouse.

    A single `Treasury` instance should be constructed once per multi-agent
    session (e.g. once per LangGraph invocation or CrewAI kickoff) and shared
    by reference across all supervisor/worker nodes.
    """

    def __init__(self, session_id: Optional[str], total_session_budget_usd: Decimal | float | str) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.total_session_budget_usd = Decimal(str(total_session_budget_usd))

        self._lock = threading.RLock()
        self._committed_usd: Decimal = Decimal("0")      # sum of ceilings of ACTIVE tokens
        self._settled_spent_usd: Decimal = Decimal("0")  # sum of actual spend from SETTLED tokens
        self._active_tokens: Dict[str, AllocationToken] = {}
        self._settled_tokens: Dict[str, AllocationToken] = {}
        self._ledger: List[TreasuryLedgerEntry] = []
        self._token_counter = itertools.count(1)

    # ------------------------------------------------------------------ #
    # Read-only accounting properties
    # ------------------------------------------------------------------ #

    @property
    def available_usd(self) -> Decimal:
        """Uncommitted, unspent capital still free to allocate."""
        with self._lock:
            return self.total_session_budget_usd - self._committed_usd - self._settled_spent_usd

    @property
    def committed_usd(self) -> Decimal:
        with self._lock:
            return self._committed_usd

    @property
    def settled_spent_usd(self) -> Decimal:
        with self._lock:
            return self._settled_spent_usd

    @property
    def utilization_ratio(self) -> Decimal:
        """
        Fraction of the total session budget that is either committed or
        already spent. Consumed by the Degradation Engine (Module C).
        """
        with self._lock:
            if self.total_session_budget_usd == 0:
                return Decimal("1")
            used = self._committed_usd + self._settled_spent_usd
            return used / self.total_session_budget_usd

    @property
    def remaining_ratio(self) -> Decimal:
        with self._lock:
            return Decimal("1") - self.utilization_ratio

    def ledger_history(self) -> List[TreasuryLedgerEntry]:
        with self._lock:
            return list(self._ledger)

    def get_token(self, token_id: str) -> AllocationToken:
        with self._lock:
            token = self._active_tokens.get(token_id) or self._settled_tokens.get(token_id)
        if token is None:
            raise UnknownAllocationTokenError(token_id)
        return token

    def active_tokens(self) -> List[AllocationToken]:
        with self._lock:
            return list(self._active_tokens.values())

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _next_token_id(self, agent_id: str) -> str:
        n = next(self._token_counter)
        return f"tok_{agent_id}_{n:06d}_{uuid.uuid4().hex[:8]}"

    def _record(
        self,
        event_type: LedgerEventType,
        agent_id: str,
        token_id: Optional[str],
        amount_usd: Decimal,
        metadata: Optional[Dict[str, str]] = None,
    ) -> TreasuryLedgerEntry:
        """Must be called while holding `self._lock`."""
        entry = TreasuryLedgerEntry(
            event_id=uuid.uuid4().hex,
            event_type=event_type,
            timestamp=time.time(),
            agent_id=agent_id,
            token_id=token_id,
            amount_usd=amount_usd,
            treasury_available_after_usd=self.total_session_budget_usd
            - self._committed_usd
            - self._settled_spent_usd,
            metadata=metadata or {},
        )
        self._ledger.append(entry)
        return entry

    # ------------------------------------------------------------------ #
    # Public synchronous core (guarded by RLock; see module docstring)
    # ------------------------------------------------------------------ #

    def _allocate_sync(
        self, agent_id: str, max_usd: Decimal, model_name: Optional[str]
    ) -> AllocationToken:
        with self._lock:
            available = self.total_session_budget_usd - self._committed_usd - self._settled_spent_usd
            if available <= Decimal("0"):
                self._record(
                    LedgerEventType.ALLOCATION_DENIED, agent_id, None, max_usd,
                    {"reason": "session_budget_exhausted"},
                )
                raise SessionBudgetExhaustedError(self.session_id)
            if max_usd > available:
                self._record(
                    LedgerEventType.ALLOCATION_DENIED, agent_id, None, max_usd,
                    {"reason": "insufficient_funds", "available": str(available)},
                )
                raise InsufficientTreasuryFundsError(max_usd, available)

            token_id = self._next_token_id(agent_id)
            token = AllocationToken(
                token_id=token_id,
                agent_id=agent_id,
                initial_ceiling_usd=max_usd,
                ceiling_usd=max_usd,
                model_name=model_name,
                created_at=time.time(),
            )
            self._active_tokens[token_id] = token
            self._committed_usd += max_usd
            self._record(LedgerEventType.ALLOCATION_GRANTED, agent_id, token_id, max_usd)
            return token

    def _release_sync(self, token: AllocationToken | str, actual_usd_spent: Decimal) -> Decimal:
        """
        Settle a token. Returns the yielded-back delta (ceiling - actual),
        which is immediately re-absorbed into `available_usd` since it is no
        longer subtracted via `_committed_usd`.
        """
        token_id = token.token_id if isinstance(token, AllocationToken) else token
        with self._lock:
            active = self._active_tokens.get(token_id)
            if active is None:
                if token_id in self._settled_tokens:
                    raise AllocationAlreadySettledError(token_id)
                raise UnknownAllocationTokenError(token_id)

            if actual_usd_spent < Decimal("0"):
                actual_usd_spent = Decimal("0")

            # Actual spend is clamped to the token's ceiling for accounting
            # purposes -- overshoot without a granted extension is a logic
            # error upstream (the guard/middleware should have prevented it)
            # but we never let the ledger itself go negative or over-commit.
            clamped_spend = min(actual_usd_spent, active.ceiling_usd)

            active.actual_spent_usd = clamped_spend
            active.status = TokenStatus.SETTLED

            self._committed_usd -= active.ceiling_usd
            self._settled_spent_usd += clamped_spend

            del self._active_tokens[token_id]
            self._settled_tokens[token_id] = active

            yielded_back = active.ceiling_usd - clamped_spend
            self._record(
                LedgerEventType.ALLOCATION_RELEASED,
                active.agent_id,
                token_id,
                clamped_spend,
                {"yielded_back_usd": str(yielded_back), "ceiling_usd": str(active.ceiling_usd)},
            )
            return yielded_back

    def _grant_extension_sync(self, token: AllocationToken | str, extra_usd: Decimal) -> AllocationToken:
        """Raise an active token's ceiling by `extra_usd`. Caller must have already
        verified against `available_usd` (the Reallocator does this)."""
        token_id = token.token_id if isinstance(token, AllocationToken) else token
        with self._lock:
            active = self._active_tokens.get(token_id)
            if active is None:
                raise UnknownAllocationTokenError(token_id)
            active.ceiling_usd += extra_usd
            active.extensions_granted_usd += extra_usd
            self._committed_usd += extra_usd
            self._record(
                LedgerEventType.CREDIT_EXTENSION_GRANTED,
                active.agent_id,
                token_id,
                extra_usd,
            )
            return active

    # ------------------------------------------------------------------ #
    # Public async-native API (required by PRD Technical Constraints)
    # ------------------------------------------------------------------ #

    async def request_allocation(
        self,
        agent_id: str,
        max_usd: Decimal | float | str,
        model_name: Optional[str] = None,
    ) -> AllocationToken:
        """
        Request a specific token/cost allocation before starting a sub-task.
        Raises `InsufficientTreasuryFundsError` or `SessionBudgetExhaustedError`
        if the central pool cannot support the request.
        """
        return self._allocate_sync(agent_id, Decimal(str(max_usd)), model_name)

    async def release_allocation(
        self, token: AllocationToken | str, actual_usd_spent: Decimal | float | str
    ) -> Decimal:
        """
        Release a token back to the treasury, settling actual spend and
        immediately yielding any unused delta back to the pool.
        Returns the yielded-back amount.
        """
        return self._release_sync(token, Decimal(str(actual_usd_spent)))

    async def grant_extension(
        self, token: AllocationToken | str, extra_usd: Decimal | float | str
    ) -> AllocationToken:
        """Low-level primitive used by `Reallocator.request_credit_extension`."""
        return self._grant_extension_sync(token, Decimal(str(extra_usd)))

    def record_extension_denial(self, agent_id: str, token_id: str, amount_usd: Decimal, reason: str) -> None:
        """
        Public audit hook used by `Reallocator` to log a denied micro-lending
        request without granting write access to the ledger's core mutation
        methods.
        """
        with self._lock:
            self._record(
                LedgerEventType.CREDIT_EXTENSION_DENIED,
                agent_id,
                token_id,
                amount_usd,
                {"reason": reason},
            )

    def snapshot(self) -> "TreasurySnapshot":
        with self._lock:
            return TreasurySnapshot(
                session_id=self.session_id,
                total_session_budget_usd=self.total_session_budget_usd,
                committed_usd=self._committed_usd,
                settled_spent_usd=self._settled_spent_usd,
                available_usd=self.total_session_budget_usd - self._committed_usd - self._settled_spent_usd,
                active_token_count=len(self._active_tokens),
                settled_token_count=len(self._settled_tokens),
            )


@dataclass(frozen=True)
class TreasurySnapshot:
    """A point-in-time, immutable read of the treasury's financial state."""

    session_id: str
    total_session_budget_usd: Decimal
    committed_usd: Decimal
    settled_spent_usd: Decimal
    available_usd: Decimal
    active_token_count: int
    settled_token_count: int
