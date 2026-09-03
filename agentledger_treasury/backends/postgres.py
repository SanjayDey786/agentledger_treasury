"""
agentledger_treasury.backends.postgres
========================================
PostgreSQL-backed Treasury for distributed AgentLedger Treasury deployments.

Multiple worker processes (across pods or hosts) can share one logical
treasury by pointing separate `PostgresTreasury` instances at the same
`session_id` and connection pool: every mutation (`request_allocation`,
`release_allocation`, `grant_extension`) is wrapped in a transaction that
takes a `SELECT ... FOR UPDATE` row lock on the session, so concurrent
calls across workers cannot oversubscribe the shared pool.

API surface note
-----------------
The in-memory `Treasury` exposes `available_usd`, `committed_usd`,
`settled_spent_usd`, `utilization_ratio`, `remaining_ratio` and
`record_extension_denial(...)` as *synchronous* members because it never
performs I/O -- and `Reallocator` / `DegradationEngine` call them
synchronously (no `await`). `PostgresTreasury` mirrors that surface with
synchronous properties backed by a small in-process cache of the shared
`sessions` row, refreshed on every allocation/release/extension this
instance performs (and via the explicit `await treasury.refresh()`).
Because other workers can mutate the same session between two of *this*
instance's operations, treat these cached properties as eventually
consistent, not linearizable -- call `refresh()` first if a decision needs
a guaranteed-fresh read. `get_token`, `snapshot`, and `ledger_history`
remain `async def` (unlike the in-memory Treasury) since they require a
real database round trip and gain nothing from faking sync-ness.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional

import asyncpg

from agentledger_treasury.core.treasury import (
    AllocationToken,
    LedgerEventType,
    TokenStatus,
    TreasuryLedgerEntry,
    TreasurySnapshot,
)
from agentledger_treasury.exceptions import (
    AllocationAlreadySettledError,
    InsufficientTreasuryFundsError,
    SessionBudgetExhaustedError,
    UnknownAllocationTokenError,
)

_CREATE_TABLES_SQL = """
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
"""


@dataclass
class _SessionState:
    total_budget_usd: Decimal
    committed_usd: Decimal
    settled_spent_usd: Decimal


class PostgresTreasury:
    """
    PostgreSQL-backed implementation of the Treasury clearinghouse.

    Expects an `asyncpg.Pool`. All mutating methods are `async` and
    transactional; the session row is locked (`SELECT ... FOR UPDATE`)
    during any mutation to guarantee consistency across workers sharing
    the same `session_id`. Tables are created lazily on first use if
    `create_tables=True` (default).
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
        self._create_tables = create_tables
        self._ready = False
        self._ready_lock = asyncio.Lock()

        # Locally cached mirror of the shared `sessions` row backing the
        # synchronous ratio properties below (see module docstring).
        self._cached_total_budget_usd: Decimal = self.total_session_budget_usd or Decimal(0)
        self._cached_committed_usd: Decimal = Decimal(0)
        self._cached_settled_spent_usd: Decimal = Decimal(0)

    # ------------------------------------------------------------------ #
    # Synchronous accounting properties (mirror Treasury; see docstring)
    # ------------------------------------------------------------------ #

    @property
    def available_usd(self) -> Decimal:
        return self._cached_total_budget_usd - self._cached_committed_usd - self._cached_settled_spent_usd

    @property
    def committed_usd(self) -> Decimal:
        return self._cached_committed_usd

    @property
    def settled_spent_usd(self) -> Decimal:
        return self._cached_settled_spent_usd

    @property
    def utilization_ratio(self) -> Decimal:
        if self._cached_total_budget_usd == 0:
            return Decimal("1")
        used = self._cached_committed_usd + self._cached_settled_spent_usd
        return used / self._cached_total_budget_usd

    @property
    def remaining_ratio(self) -> Decimal:
        return Decimal("1") - self.utilization_ratio

    def _update_cache(self, state: _SessionState) -> None:
        self._cached_total_budget_usd = state.total_budget_usd
        self._cached_committed_usd = state.committed_usd
        self._cached_settled_spent_usd = state.settled_spent_usd

    async def refresh(self) -> None:
        """Force-refresh the locally cached snapshot from Postgres."""
        await self._ensure_ready()
        async with self.pool.acquire() as conn:
            self._update_cache(await self._session_state(conn, for_update=False))

    # ------------------------------------------------------------------ #
    # Lazy bootstrap / session row helpers
    # ------------------------------------------------------------------ #

    async def _ensure_ready(self) -> None:
        """Create tables on first use. Idempotent, safe to call repeatedly."""
        if self._ready or not self._create_tables:
            self._ready = True
            return
        async with self._ready_lock:
            if self._ready:
                return
            async with self.pool.acquire() as conn:
                # A Postgres advisory lock serializes schema creation across
                # concurrent *processes* too -- the asyncio.Lock above only
                # protects concurrent calls within this one process. Without
                # it, two independent workers racing to run
                # `CREATE TABLE IF NOT EXISTS` for the first time can both
                # pass Postgres's existence check before either commits,
                # producing a duplicate-key error against the catalog.
                await conn.execute("SELECT pg_advisory_lock(727477347)")
                try:
                    await conn.execute(_CREATE_TABLES_SQL)
                finally:
                    await conn.execute("SELECT pg_advisory_unlock(727477347)")
            self._ready = True

    async def _session_state(self, conn: asyncpg.Connection, *, for_update: bool) -> _SessionState:
        """
        Read the session row, optionally taking a `FOR UPDATE` lock. If the
        session does not exist yet and `for_update=True`, it is atomically
        created (idempotent even if two workers race to create it).
        """
        suffix = " FOR UPDATE" if for_update else ""
        row = await conn.fetchrow(
            f"SELECT total_budget_usd, committed_usd, settled_spent_usd FROM sessions "
            f"WHERE session_id = $1{suffix}",
            self.session_id,
        )
        if row is None:
            if self.total_session_budget_usd is None:
                raise ValueError(
                    "total_session_budget_usd must be provided when creating a new session"
                )
            if not for_update:
                return _SessionState(self.total_session_budget_usd, Decimal(0), Decimal(0))
            row = await conn.fetchrow(
                """
                INSERT INTO sessions (session_id, total_budget_usd)
                VALUES ($1, $2)
                ON CONFLICT (session_id) DO UPDATE SET session_id = EXCLUDED.session_id
                RETURNING total_budget_usd, committed_usd, settled_spent_usd
                """,
                self.session_id,
                self.total_session_budget_usd,
            )
        return _SessionState(row["total_budget_usd"], row["committed_usd"], row["settled_spent_usd"])

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
            datetime.now(timezone.utc),
            agent_id,
            token_id,
            amount_usd,
            available_after_usd,
            json.dumps(metadata or {}),
        )

    # ------------------------------------------------------------------ #
    # Public async API (mirrors Treasury; see module docstring)
    # ------------------------------------------------------------------ #

    async def request_allocation(
        self,
        agent_id: str,
        max_usd: Decimal | float | str,
        model_name: Optional[str] = None,
    ) -> AllocationToken:
        await self._ensure_ready()
        max_usd_dec = Decimal(str(max_usd))
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                state = await self._session_state(conn, for_update=True)
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
                    self._update_cache(state)
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
                    self._update_cache(state)
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
                self._update_cache(_SessionState(state.total_budget_usd, new_committed, state.settled_spent_usd))
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
        await self._ensure_ready()
        token_id = token.token_id if isinstance(token, AllocationToken) else token
        actual_dec = Decimal(str(actual_usd_spent))
        if actual_dec < Decimal(0):
            actual_dec = Decimal(0)

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                state = await self._session_state(conn, for_update=True)

                row = await conn.fetchrow(
                    "SELECT * FROM allocations WHERE token_id = $1 AND session_id = $2 FOR UPDATE",
                    token_id,
                    self.session_id,
                )
                if row is None:
                    raise UnknownAllocationTokenError(token_id)
                if row["status"] == TokenStatus.SETTLED.value:
                    raise AllocationAlreadySettledError(token_id)

                ceiling: Decimal = row["ceiling_usd"]
                clamped_spend: Decimal = min(actual_dec, ceiling)
                yielded_back: Decimal = ceiling - clamped_spend

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
                self._update_cache(_SessionState(state.total_budget_usd, new_committed, new_settled))
                return yielded_back

    async def grant_extension(
        self, token: AllocationToken | str, extra_usd: Decimal | float | str
    ) -> AllocationToken:
        await self._ensure_ready()
        token_id = token.token_id if isinstance(token, AllocationToken) else token
        extra_dec = Decimal(str(extra_usd))

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                state = await self._session_state(conn, for_update=True)

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
                self._update_cache(_SessionState(state.total_budget_usd, new_committed, state.settled_spent_usd))

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
        """
        Synchronous audit hook (matches `Treasury.record_extension_denial`)
        used by `Reallocator`, which calls it without `await` from inside
        an already-running event loop. We fire the DB write as a
        best-effort background task rather than blocking synchronously --
        losing an audit-log row on abrupt process exit is an acceptable
        tradeoff for keeping this a non-blocking, sync-compatible call.
        """

        async def _log() -> None:
            await self._ensure_ready()
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    state = await self._session_state(conn, for_update=False)
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

        try:
            asyncio.get_running_loop()
            asyncio.create_task(_log())
        except RuntimeError:
            asyncio.run(_log())

    async def snapshot(self) -> TreasurySnapshot:
        """
        Unlike `Treasury.snapshot`, this is `async` since it requires a
        database round trip.
        """
        await self._ensure_ready()
        async with self.pool.acquire() as conn:
            state = await self._session_state(conn, for_update=False)
            self._update_cache(state)
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

    async def get_token(self, token_id: str) -> AllocationToken:
        """
        Unlike `Treasury.get_token`, this is `async` since it requires a
        database round trip.
        """
        await self._ensure_ready()
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

    async def ledger_history(self) -> List[TreasuryLedgerEntry]:
        """
        Unlike `Treasury.ledger_history`, this is `async` since it requires
        a database round trip.
        """
        await self._ensure_ready()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM ledger_entries WHERE session_id = $1 ORDER BY timestamp ASC",
                self.session_id,
            )
            return [
                TreasuryLedgerEntry(
                    event_id=row["event_id"],
                    event_type=LedgerEventType(row["event_type"]),
                    timestamp=row["timestamp"].timestamp(),
                    agent_id=row["agent_id"],
                    token_id=row["token_id"],
                    amount_usd=row["amount_usd"],
                    treasury_available_after_usd=row["available_after_usd"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )
                for row in rows
            ]
