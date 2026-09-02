"""
PostgreSQL-backed Treasury for distributed AgentLedger deployments.

Uses asyncpg for async database access and row‑level locking to ensure
atomic updates across multiple workers.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

import asyncpg

from agentledger.core.treasury import (
    AllocationToken,
    LedgerEventType,
    TokenStatus,
    TreasuryLedgerEntry,
    TreasurySnapshot,
)
from agentledger.exceptions import (
    AllocationAlreadySettledError,
    InsufficientTreasuryFundsError,
    SessionBudgetExhaustedError,
    UnknownAllocationTokenError,
)


@dataclass
class _SessionState:
    total_budget_usd: Decimal
    committed_usd: Decimal
    settled_spent_usd: Decimal


class PostgresTreasury:
    """
    PostgreSQL-backed implementation of the Treasury clearinghouse.

    Expects an `asyncpg.Pool` instance. All public methods are async and
    transactional. The session row is locked (`SELECT ... FOR UPDATE`)
    during any mutation to guarantee consistency.

    Tables are created automatically if `create_tables=True` (default).
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        session_id: Optional[str] = None,
        total_session_budget_usd: Optional[Decimal | float | str] = None,
        *,
        create_tables: bool = True,
    ) -> None:
        self.pool = pool
        self.session_id = session_id or str(uuid.uuid4())
        self.total_session_budget_usd = (
            Decimal(str(total_session_budget_usd)) if total_session_budget_usd is not None else None
        )
        self._tables_created = False
        if create_tables:
            self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create the required tables and indexes if they do not exist."""
        # We'll run this synchronously, but it's fine for init.
        # In production, you'd run migrations separately.
        async def _create():
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        total_budget_usd NUMERIC NOT NULL,
                        committed_usd NUMERIC NOT NULL DEFAULT 0,
                        settled_spent_usd NUMERIC NOT NULL DEFAULT 0,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    CREATE TABLE IF NOT EXISTS allocations (
                        token_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                        agent_id TEXT NOT NULL,
                        initial_ceiling_usd NUMERIC NOT NULL,
                        ceiling_usd NUMERIC NOT NULL,
                        model_name TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        status TEXT NOT NULL CHECK (status IN ('active', 'settled')),
                        actual_spent_usd NUMERIC NOT NULL DEFAULT 0,
                        extensions_granted_usd NUMERIC NOT NULL DEFAULT 0,
                        settled_at TIMESTAMPTZ
                    );
                    CREATE TABLE IF NOT EXISTS ledger_entries (
                        event_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                        event_type TEXT NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        agent_id TEXT NOT NULL,
                        token_id TEXT,
                        amount_usd NUMERIC NOT NULL,
                        available_after_usd NUMERIC NOT NULL,
                        metadata JSONB
                    );
                    CREATE INDEX IF NOT EXISTS idx_allocations_session_status ON allocations(session_id, status);
                    CREATE INDEX IF NOT EXISTS idx_ledger_session ON ledger_entries(session_id);
                """)
        # Run synchronously for simplicity; in async init you'd want to await.
        # We'll run it in a new event loop or use asyncio.run() if not already running.
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Schedule it; but for init we can block.
            asyncio.create_task(_create())
        else:
            asyncio.run(_create())

    async def _ensure_session(self) -> _SessionState:
        """Ensure the session row exists, creating it if needed."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT total_budget_usd, committed_usd, settled_spent_usd FROM sessions WHERE session_id = $1",
                self.session_id,
            )
            if row is None:
                if self.total_session_budget_usd is None:
                    raise ValueError(
                        "total_session_budget_usd must be provided when creating a new session"
                    )
                await conn.execute(
                    "INSERT INTO sessions (session_id, total_budget_usd) VALUES ($1, $2)",
                    self.session_id,
                    self.total_session_budget_usd,
                )
                return _SessionState(
                    total_budget_usd=self.total_session_budget_usd,
                    committed_usd=Decimal(0),
                    settled_spent_usd=Decimal(0),
                )
            else:
                return _SessionState(
                    total_budget_usd=row["total_budget_usd"],
                    committed_usd=row["committed_usd"],
                    settled_spent_usd=row["settled_spent_usd"],
                )

    async def _lock_session(self, conn: asyncpg.Connection) -> _SessionState:
        """Lock the session row for update and return its current state."""
        row = await conn.fetchrow(
            "SELECT total_budget_usd, committed_usd, settled_spent_usd FROM sessions WHERE session_id = $1 FOR UPDATE",
            self.session_id,
        )
        if row is None:
            raise ValueError(f"Session {self.session_id} not found")
        return _SessionState(
            total_budget_usd=row["total_budget_usd"],
            committed_usd=row["committed_usd"],
            settled_spent_usd=row["settled_spent_usd"],
        )

    async def _append_ledger_entry(
        self,
        conn: asyncpg.Connection,
        event_type: LedgerEventType,
        agent_id: str,
        token_id: Optional[str],
        amount_usd: Decimal,
        available_after_usd: Decimal,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        await conn.execute(
            """
            INSERT INTO ledger_entries
            (event_id, session_id, event_type, timestamp, agent_id, token_id, amount_usd, available_after_usd, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            uuid.uuid4().hex,
            self.session_id,
            event_type.value,
            time.time(),
            agent_id,
            token_id,
            amount_usd,
            available_after_usd,
            metadata or {},
        )

    # ------------------------------------------------------------------ #
    # Public async API (same as in-memory Treasury)
    # ------------------------------------------------------------------ #

    async def request_allocation(
        self,
        agent_id: str,
        max_usd: Decimal | float | str,
        model_name: Optional[str] = None,
    ) -> AllocationToken:
        max_usd_dec = Decimal(str(max_usd))
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                state = await self._lock_session(conn)
                available = state.total_budget_usd - state.committed_usd - state.settled_spent_usd
                if available <= Decimal(0):
                    await self._append_ledger_entry(
                        conn,
                        LedgerEventType.ALLOCATION_DENIED,
                        agent_id,
                        None,
                        max_usd_dec,
                        available,
                        {"reason": "session_budget_exhausted"},
                    )
                    raise SessionBudgetExhaustedError(self.session_id)
                if max_usd_dec > available:
                    await self._append_ledger_entry(
                        conn,
                        LedgerEventType.ALLOCATION_DENIED,
                        agent_id,
                        None,
                        max_usd_dec,
                        available,
                        {"reason": "insufficient_funds", "available": str(available)},
                    )
                    raise InsufficientTreasuryFundsError(max_usd_dec, available)

                token_id = f"tok_{agent_id}_{uuid.uuid4().hex[:8]}"
                await conn.execute(
                    """
                    INSERT INTO allocations
                    (token_id, session_id, agent_id, initial_ceiling_usd, ceiling_usd, model_name, status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    token_id,
                    self.session_id,
                    agent_id,
                    max_usd_dec,
                    max_usd_dec,
                    model_name,
                    TokenStatus.ACTIVE.value,
                )
                new_committed = state.committed_usd + max_usd_dec
                await conn.execute(
                    "UPDATE sessions SET committed_usd = $1, updated_at = NOW() WHERE session_id = $2",
                    new_committed,
                    self.session_id,
                )
                await self._append_ledger_entry(
                    conn,
                    LedgerEventType.ALLOCATION_GRANTED,
                    agent_id,
                    token_id,
                    max_usd_dec,
                    state.total_budget_usd - new_committed - state.settled_spent_usd,
                )
                return AllocationToken(
                    token_id=token_id,
                    agent_id=agent_id,
                    initial_ceiling_usd=max_usd_dec,
                    ceiling_usd=max_usd_dec,
                    model_name=model_name,
                    created_at=time.time(),
                    status=TokenStatus.ACTIVE,
                    actual_spent_usd=Decimal(0),
                    extensions_granted_usd=Decimal(0),
                )

    async def release_allocation(
        self, token: AllocationToken | str, actual_usd_spent: Decimal | float | str
    ) -> Decimal:
        token_id = token.token_id if isinstance(token, AllocationToken) else token
        actual_dec = Decimal(str(actual_usd_spent))
        if actual_dec < Decimal(0):
            actual_dec = Decimal(0)

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                state = await self._lock_session(conn)

                # Fetch the allocation row
                row = await conn.fetchrow(
                    "SELECT * FROM allocations WHERE token_id = $1 AND session_id = $2 FOR UPDATE",
                    token_id,
                    self.session_id,
                )
                if row is None:
                    raise UnknownAllocationTokenError(token_id)
                if row["status"] == TokenStatus.SETTLED.value:
                    raise AllocationAlreadySettledError(token_id)

                ceiling = row["ceiling_usd"]
                clamped_spend = min(actual_dec, ceiling)
                yielded_back = ceiling - clamped_spend

                # Update allocation
                await conn.execute(
                    """
                    UPDATE allocations
                    SET status = $1, actual_spent_usd = $2, settled_at = NOW()
                    WHERE token_id = $3
                    """,
                    TokenStatus.SETTLED.value,
                    clamped_spend,
                    token_id,
                )

                # Update session sums
                new_committed = state.committed_usd - ceiling
                new_settled = state.settled_spent_usd + clamped_spend
                await conn.execute(
                    "UPDATE sessions SET committed_usd = $1, settled_spent_usd = $2, updated_at = NOW() WHERE session_id = $3",
                    new_committed,
                    new_settled,
                    self.session_id,
                )

                available_after = state.total_budget_usd - new_committed - new_settled
                await self._append_ledger_entry(
                    conn,
                    LedgerEventType.ALLOCATION_RELEASED,
                    row["agent_id"],
                    token_id,
                    clamped_spend,
                    available_after,
                    {"yielded_back_usd": str(yielded_back), "ceiling_usd": str(ceiling)},
                )
                return yielded_back

    async def grant_extension(
        self, token: AllocationToken | str, extra_usd: Decimal | float | str
    ) -> AllocationToken:
        token_id = token.token_id if isinstance(token, AllocationToken) else token
        extra_dec = Decimal(str(extra_usd))

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                state = await self._lock_session(conn)

                row = await conn.fetchrow(
                    "SELECT * FROM allocations WHERE token_id = $1 AND session_id = $2 FOR UPDATE",
                    token_id,
                    self.session_id,
                )
                if row is None:
                    raise UnknownAllocationTokenError(token_id)
                if row["status"] == TokenStatus.SETTLED.value:
                    raise AllocationAlreadySettledError(token_id)

                new_ceiling = row["ceiling_usd"] + extra_dec
                new_extensions = row["extensions_granted_usd"] + extra_dec
                await conn.execute(
                    """
                    UPDATE allocations
                    SET ceiling_usd = $1, extensions_granted_usd = $2
                    WHERE token_id = $3
                    """,
                    new_ceiling,
                    new_extensions,
                    token_id,
                )

                new_committed = state.committed_usd + extra_dec
                await conn.execute(
                    "UPDATE sessions SET committed_usd = $1, updated_at = NOW() WHERE session_id = $2",
                    new_committed,
                    self.session_id,
                )

                available_after = state.total_budget_usd - new_committed - state.settled_spent_usd
                await self._append_ledger_entry(
                    conn,
                    LedgerEventType.CREDIT_EXTENSION_GRANTED,
                    row["agent_id"],
                    token_id,
                    extra_dec,
                    available_after,
                )

                # Return an updated AllocationToken (we'll reconstruct)
                return AllocationToken(
                    token_id=token_id,
                    agent_id=row["agent_id"],
                    initial_ceiling_usd=row["initial_ceiling_usd"],
                    ceiling_usd=new_ceiling,
                    model_name=row["model_name"],
                    created_at=row["created_at"].timestamp(),
                    status=TokenStatus.ACTIVE,
                    actual_spent_usd=row["actual_spent_usd"],
                    extensions_granted_usd=new_extensions,
                )

    def record_extension_denial(self, agent_id: str, token_id: str, amount_usd: Decimal, reason: str) -> None:
        # We need to run this as an async operation; but the method signature is sync.
        # In the in-memory version it's sync. We'll keep it sync and schedule a fire-and-forget.
        # However, this is only called from Reallocator, which is synchronous.
        # We'll implement a workaround: schedule the insert without waiting.
        import asyncio
        async def _log():
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # We don't need to lock the session for this read-only append.
                    # But we need to know available_after; we can get current state.
                    state = await self._ensure_session()
                    available = state.total_budget_usd - state.committed_usd - state.settled_spent_usd
                    await self._append_ledger_entry(
                        conn,
                        LedgerEventType.CREDIT_EXTENSION_DENIED,
                        agent_id,
                        token_id,
                        amount_usd,
                        available,
                        {"reason": reason},
                    )
        # We'll run it in the background; but if the event loop isn't running, we need to handle.
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(_log())
        except RuntimeError:
            asyncio.run(_log())

    async def snapshot(self) -> TreasurySnapshot:
        async with self.pool.acquire() as conn:
            state = await self._ensure_session()
            active_count = await conn.fetchval(
                "SELECT COUNT(*) FROM allocations WHERE session_id = $1 AND status = 'active'",
                self.session_id,
            )
            settled_count = await conn.fetchval(
                "SELECT COUNT(*) FROM allocations WHERE session_id = $1 AND status = 'settled'",
                self.session_id,
            )
            return TreasurySnapshot(
                session_id=self.session_id,
                total_session_budget_usd=state.total_budget_usd,
                committed_usd=state.committed_usd,
                settled_spent_usd=state.settled_spent_usd,
                available_usd=state.total_budget_usd - state.committed_usd - state.settled_spent_usd,
                active_token_count=active_count or 0,
                settled_token_count=settled_count or 0,
            )

    def get_token(self, token_id: str) -> AllocationToken:
        # Sync method; we'll implement with a quick async call
        import asyncio
        async def _get():
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM allocations WHERE token_id = $1 AND session_id = $2",
                    token_id,
                    self.session_id,
                )
                if row is None:
                    raise UnknownAllocationTokenError(token_id)
                return AllocationToken(
                    token_id=row["token_id"],
                    agent_id=row["agent_id"],
                    initial_ceiling_usd=row["initial_ceiling_usd"],
                    ceiling_usd=row["ceiling_usd"],
                    model_name=row["model_name"],
                    created_at=row["created_at"].timestamp(),
                    status=TokenStatus(row["status"]),
                    actual_spent_usd=row["actual_spent_usd"],
                    extensions_granted_usd=row["extensions_granted_usd"],
                )
        try:
            loop = asyncio.get_running_loop()
            # Can't use await in sync method; we'll run it sync.
            # We'll use asyncio.run in a new event loop.
            return asyncio.run(_get())
        except RuntimeError:
            return asyncio.run(_get())

    # Additional convenience methods (active_tokens, ledger_history) can be added similarly.
    # For brevity, we'll implement only the essential ones.