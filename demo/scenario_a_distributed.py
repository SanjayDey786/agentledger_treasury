"""
Scenario A -- what a distributed backend actually buys you.

Two sub-demos, both using REAL separate OS processes (not asyncio tasks in
one process -- that would only prove single-process concurrency safety,
which the in-memory Treasury already has via its RLock). This is the
"useful in some case" half of the demo: coordinating spend across multiple
independent workers is a problem the in-memory Treasury cannot solve at
all, by construction.

A1: Oversubscription. N worker processes each try to draw from a shared
    $1.00 budget.
      - "naive" baseline: each process has its OWN in-memory Treasury,
        independently initialized with total_session_budget_usd="1.00".
        This is what teams do by default when they haven't set up a
        distributed backend: each pod/worker tracks its own local
        counter. There is no way for process 2 to know process 1 already
        spent money, because nothing is shared.
      - "coordinated": all N processes point PostgresTreasury at the same
        session_id + Postgres instance. Every allocation takes a real
        `SELECT ... FOR UPDATE` row lock, so processes are serialized
        against each other even though they share no memory.

A2: Durability. A process allocates money against a Postgres-backed
    session, then exits (simulating a crash) BEFORE settling. A brand new
    process, with no memory of the first one, opens the same session_id
    and observes the exact correct remaining budget. An in-memory Treasury
    cannot do this even in principle -- its state dies with the process.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentledger_treasury.core.treasury import Treasury  # noqa: E402
from agentledger_treasury.exceptions import (  # noqa: E402
    InsufficientTreasuryFundsError,
    SessionBudgetExhaustedError,
)

TOTAL_BUDGET = "1.00"
PER_WORKER_ATTEMPTS = 5
PER_ATTEMPT_USD = "0.30"  # 4 workers x 5 attempts x $0.30 = $6.00 demanded against a $1.00 budget
N_WORKERS = 4


# ------------------------------------------------------------------ #
# A1: naive baseline -- each process has its own private Treasury
# ------------------------------------------------------------------ #

def _naive_worker(worker_id: int, out_path: str) -> None:
    async def run() -> None:
        treasury = Treasury(session_id=f"naive-{worker_id}", total_session_budget_usd=TOTAL_BUDGET)
        granted = Decimal("0")
        for _ in range(PER_WORKER_ATTEMPTS):
            try:
                await treasury.request_allocation(f"worker-{worker_id}", PER_ATTEMPT_USD)
                granted += Decimal(PER_ATTEMPT_USD)
            except (InsufficientTreasuryFundsError, SessionBudgetExhaustedError):
                pass
        with open(out_path, "w") as f:
            json.dump({"worker_id": worker_id, "granted_usd": str(granted)}, f)

    asyncio.run(run())


# ------------------------------------------------------------------ #
# A1: coordinated -- all processes share one Postgres-backed session
# ------------------------------------------------------------------ #

def _postgres_worker(worker_id: int, session_id: str, out_path: str) -> None:
    async def run() -> None:
        from pg import make_pool
        from agentledger_treasury.backends.postgres import PostgresTreasury

        pool = await make_pool()
        treasury = PostgresTreasury(
            pool, session_id=session_id, total_session_budget_usd=TOTAL_BUDGET, create_tables=True
        )
        granted = Decimal("0")
        for _ in range(PER_WORKER_ATTEMPTS):
            try:
                await treasury.request_allocation(f"worker-{worker_id}", PER_ATTEMPT_USD)
                granted += Decimal(PER_ATTEMPT_USD)
            except (InsufficientTreasuryFundsError, SessionBudgetExhaustedError):
                pass
        await pool.close()
        with open(out_path, "w") as f:
            json.dump({"worker_id": worker_id, "granted_usd": str(granted)}, f)

    asyncio.run(run())


def _run_workers(target, args_list, tmp_dir: str, *, tag: str) -> list[dict]:
    procs = []
    out_paths = []
    for i, args in enumerate(args_list):
        # `tag` keeps filenames unique per call so a crashed worker that
        # never writes its file can't be masked by a stale file left over
        # from a previous batch that happened to use the same path.
        out_path = os.path.join(tmp_dir, f"result_{tag}_{i}.json")
        if os.path.exists(out_path):
            os.remove(out_path)
        out_paths.append(out_path)
        p = mp.Process(target=target, args=(*args, out_path))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"worker process {p.pid} exited with code {p.exitcode} -- see traceback above")
    results = []
    for path in out_paths:
        with open(path) as f:
            results.append(json.load(f))
    return results


def demo_a1(tmp_dir: str) -> None:
    print("=" * 70)
    print("A1 -- Oversubscription across independent processes")
    print(f"Nominal shared budget: ${TOTAL_BUDGET}  |  {N_WORKERS} worker processes each")
    print(f"attempt {PER_WORKER_ATTEMPTS}x ${PER_ATTEMPT_USD} allocations (${Decimal(PER_ATTEMPT_USD) * PER_WORKER_ATTEMPTS * N_WORKERS} demanded total)")
    print("=" * 70)

    print("\n[naive] each process has its own private in-memory Treasury...")
    naive_results = _run_workers(_naive_worker, [(i,) for i in range(N_WORKERS)], tmp_dir, tag="naive")
    naive_total = sum(Decimal(r["granted_usd"]) for r in naive_results)
    for r in naive_results:
        print(f"  worker-{r['worker_id']}: granted ${r['granted_usd']}")
    print(f"  TOTAL GRANTED across all workers: ${naive_total}  (nominal budget was only ${TOTAL_BUDGET})")
    if naive_total > Decimal(TOTAL_BUDGET):
        print(f"  >>> OVERSUBSCRIBED by ${naive_total - Decimal(TOTAL_BUDGET)} -- each worker thought it had the full budget to itself.")

    print("\n[coordinated] all processes share one PostgresTreasury session...")
    session_id = "a1-coordinated-demo"
    pg_results = _run_workers(
        _postgres_worker, [(i, session_id) for i in range(N_WORKERS)], tmp_dir, tag="pg"
    )
    pg_total = sum(Decimal(r["granted_usd"]) for r in pg_results)
    for r in pg_results:
        print(f"  worker-{r['worker_id']}: granted ${r['granted_usd']}")
    print(f"  TOTAL GRANTED across all workers: ${pg_total}  (nominal budget was ${TOTAL_BUDGET})")
    if pg_total <= Decimal(TOTAL_BUDGET):
        print(f"  >>> HELD THE LINE: {N_WORKERS} independent processes never granted more than the shared budget.")
    else:
        print(f"  >>> FAILED: oversubscribed by ${pg_total - Decimal(TOTAL_BUDGET)} -- this should never happen.")


# ------------------------------------------------------------------ #
# A2: durability across a process crash
# ------------------------------------------------------------------ #

def _crash_before_settling(session_id: str, out_path: str) -> None:
    async def run() -> None:
        from pg import make_pool
        from agentledger_treasury.backends.postgres import PostgresTreasury

        pool = await make_pool()
        treasury = PostgresTreasury(
            pool, session_id=session_id, total_session_budget_usd="1.00", create_tables=True
        )
        token = await treasury.request_allocation("agent-a", "0.40")
        with open(out_path, "w") as f:
            json.dump({"token_id": token.token_id}, f)
        await pool.close()
        # process exits here WITHOUT calling release_allocation --
        # simulating a crash mid-task.

    asyncio.run(run())


def _resume_after_crash(session_id: str, out_path: str) -> None:
    async def run() -> None:
        from pg import make_pool
        from agentledger_treasury.backends.postgres import PostgresTreasury

        pool = await make_pool()
        # A brand new process, brand new PostgresTreasury instance, with NO
        # in-memory knowledge of what the crashed process did.
        treasury = PostgresTreasury(pool, session_id=session_id, create_tables=False)
        snap = await treasury.snapshot()
        await pool.close()
        with open(out_path, "w") as f:
            json.dump(
                {
                    "available_usd": str(snap.available_usd),
                    "committed_usd": str(snap.committed_usd),
                    "active_token_count": snap.active_token_count,
                },
                f,
            )

    asyncio.run(run())


def demo_a2(tmp_dir: str) -> None:
    print("\n" + "=" * 70)
    print("A2 -- Durability: a process crashes after allocating, before settling")
    print("=" * 70)

    session_id = "a2-durability-demo"
    out1 = os.path.join(tmp_dir, "crash_out.json")
    p1 = mp.Process(target=_crash_before_settling, args=(session_id, out1))
    p1.start()
    p1.join()
    if p1.exitcode != 0:
        raise RuntimeError(f"worker process {p1.pid} exited with code {p1.exitcode} -- see traceback above")
    with open(out1) as f:
        crash_info = json.load(f)
    print(f"[process 1] allocated $0.40 (token {crash_info['token_id']}), then exited -- simulating a crash.")
    print("            An in-memory Treasury's entire state would be gone right now.")

    out2 = os.path.join(tmp_dir, "resume_out.json")
    p2 = mp.Process(target=_resume_after_crash, args=(session_id, out2))
    p2.start()
    p2.join()
    if p2.exitcode != 0:
        raise RuntimeError(f"worker process {p2.pid} exited with code {p2.exitcode} -- see traceback above")
    with open(out2) as f:
        resumed = json.load(f)
    print(f"[process 2] brand new process, same session_id, reads: {resumed}")
    if resumed["active_token_count"] == 1 and resumed["available_usd"] == "0.60":
        print("  >>> State survived the crash intact: $0.60 available, 1 active (unsettled) token.")
    else:
        print("  >>> Unexpected state -- see values above.")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        demo_a1(tmp_dir)
        demo_a2(tmp_dir)
