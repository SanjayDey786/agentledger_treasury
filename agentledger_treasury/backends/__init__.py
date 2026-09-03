"""Backend storage implementations for AgentLedger Treasury.

`PostgresTreasury` is imported lazily so that `agentledger_treasury` remains
importable without the optional `postgres` extra (`asyncpg`) installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentledger_treasury.backends.postgres import PostgresTreasury

__all__ = ["PostgresTreasury"]


def __getattr__(name: str) -> object:
    if name == "PostgresTreasury":
        try:
            from agentledger_treasury.backends.postgres import PostgresTreasury
        except ImportError as exc:
            raise ImportError(
                "PostgresTreasury requires the 'postgres' extra. Install it with: "
                'pip install "agentledger_treasury[postgres]"'
            ) from exc
        return PostgresTreasury
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
