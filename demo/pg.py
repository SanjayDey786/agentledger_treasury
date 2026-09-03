"""Shared Postgres connection helper for the demo scripts."""

from __future__ import annotations

import os

DSN = os.environ.get("DEMO_POSTGRES_DSN", "postgresql://demo:demo@localhost:55432/agentledger_demo")


async def make_pool():
    import asyncpg

    return await asyncpg.create_pool(dsn=DSN, min_size=1, max_size=10)


async def reset_session(session_id: str) -> None:
    """Wipe any prior state for a session_id so a demo run starts clean."""
    import asyncpg

    conn = await asyncpg.connect(dsn=DSN)
    try:
        await conn.execute(
            """
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
            """
        )
        await conn.execute("DELETE FROM ledger_entries WHERE session_id = $1", session_id)
        await conn.execute("DELETE FROM allocations WHERE session_id = $1", session_id)
        await conn.execute("DELETE FROM sessions WHERE session_id = $1", session_id)
    finally:
        await conn.close()
