"""Process-level parallelism for independent Thomson fits.

Why processes (not threads): on a CPU box a single JAX/XLA fit uses only a
handful of cores (measured ~3-4 of 48 on the dev machine), and the Python
optimizer loop holds the GIL, so threads buy no real parallelism. Independent
fits — L-curve sweep points, multi-shot batches — are therefore run in separate
processes to fill the idle cores. Each fit keeps its own XLA runtime; results
are bit-identical to running the fits one at a time.

The ``spawn`` start method is mandatory: JAX must not be ``fork()``-ed after it
has initialized, or the child can deadlock/segfault. Each worker is optionally
pinned to a disjoint block of CPU cores (Linux ``sched_setaffinity``) so N
workers don't oversubscribe the box during the (thread-hungry) compile phase.
"""
from __future__ import annotations

import multiprocessing as _mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def serial_requested():
    """True if the user forced everything serial via THOMSON_NO_PARALLEL.

    Set ``THOMSON_NO_PARALLEL=1`` (or the ``--serial`` CLI flag, which sets it)
    to disable both the L-curve process pool and intra-fit device sharding —
    e.g. on a small / shared machine where you want to keep core usage low.
    Anything other than unset / 0 / false / no (case-insensitive) counts as on.
    """
    return os.environ.get("THOMSON_NO_PARALLEL", "").strip().lower() not in (
        "", "0", "false", "no",
    )


def available_cores():
    """Number of CPU cores this process may actually use."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except OSError:
            pass
    return os.cpu_count() or 1


def default_n_workers(n_tasks, *, cores_per_worker=4, cap=None):
    """Pick a worker count that fills the box without oversubscribing.

    A single fit uses ~3-4 cores, so ``cores_per_worker=4`` packs ~12 fits on a
    48-core box. Never more workers than tasks. Returns 1 (fully serial) when
    THOMSON_NO_PARALLEL is set.
    """
    if serial_requested():
        return 1
    budget = max(1, available_cores() // max(1, int(cores_per_worker)))
    n = min(int(n_tasks), budget)
    if cap is not None:
        n = min(n, int(cap))
    return max(1, n)


def _worker_init(counter, lock, n_workers, blas_threads):
    """Per-worker setup: cap BLAS fan-out and pin to a disjoint core block."""
    # Limit BLAS thread fan-out (numpy/scipy in data prep) so it doesn't fight
    # XLA for the worker's cores.
    if blas_threads is not None:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(var, str(int(blas_threads)))

    # Pin this worker to its own slice of cores. Pinning before the first heavy
    # XLA/BLAS call (i.e. here, in the initializer) is what actually caps each
    # worker's thread pool to the assigned block.
    if counter is not None and hasattr(os, "sched_setaffinity"):
        with lock:
            idx = counter.value
            counter.value += 1
        try:
            cores = sorted(os.sched_getaffinity(0))
        except OSError:
            cores = list(range(os.cpu_count() or 1))
        k = max(1, len(cores) // max(1, n_workers))
        block = cores[idx * k:(idx + 1) * k] or [cores[idx % len(cores)]]
        try:
            os.sched_setaffinity(0, set(block))
        except OSError:
            pass


def parallel_map(func, tasks, *, n_workers=None, cores_per_worker=4,
                 pin=True, blas_threads=None, on_complete=None):
    """Apply ``func`` to each task in a separate spawned process.

    ``func`` must be a top-level (importable, picklable) callable and every
    element of ``tasks`` must be picklable. Results are returned in task order.
    Falls back to a plain serial map when only one worker/task is involved.

    Parameters
    ----------
    func : callable
        ``func(task) -> result``. Must be importable by qualified name (spawn
        pickles it by reference), so define it at module top level.
    tasks : iterable
        Picklable task objects, one per ``func`` call.
    n_workers : int or None
        Number of worker processes. ``None`` → :func:`default_n_workers`.
        ``<= 1`` runs serially in this process.
    cores_per_worker : int
        Used both to size ``default_n_workers`` and (with ``pin``) to bound each
        worker's BLAS thread count when ``blas_threads`` is None.
    pin : bool
        Pin each worker to a disjoint block of CPU cores (Linux only).
    blas_threads : int or None
        BLAS threads per worker. ``None`` → ``cores_per_worker``.
    on_complete : callable or None
        Optional ``on_complete(index, result)`` called as each task finishes
        (in completion order, not task order) — e.g. to print progress. Fired in
        every path, including the serial fallbacks.
    """
    def _serial():
        out = []
        for i, t in enumerate(tasks):
            r = func(t)
            out.append(r)
            if on_complete is not None:
                on_complete(i, r)
        return out

    tasks = list(tasks)
    if not tasks:
        return []
    if serial_requested():                       # global kill-switch
        return _serial()
    if n_workers is None:
        n_workers = default_n_workers(len(tasks), cores_per_worker=cores_per_worker)
    if n_workers <= 1 or len(tasks) == 1:
        return _serial()
    if blas_threads is None:
        blas_threads = cores_per_worker

    ctx = _mp.get_context("spawn")

    # Disjoint core-block assignment needs a counter that is genuinely *shared*
    # across the spawned workers. A raw ctx.Value passed through initargs is
    # pickled by value under spawn (each worker gets a private copy that reads
    # 0), so use a Manager proxy, which is shared by reference.
    do_pin = pin and hasattr(os, "sched_setaffinity")
    manager = ctx.Manager() if do_pin else None
    counter = manager.Value("i", 0) if do_pin else None
    lock = manager.Lock() if do_pin else None

    results = [None] * len(tasks)
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx,
            initializer=_worker_init,
            initargs=(counter, lock, n_workers, blas_threads),
        ) as ex:
            fut_to_idx = {ex.submit(func, t): i for i, t in enumerate(tasks)}
            for fut in as_completed(fut_to_idx):
                i = fut_to_idx[fut]
                res = fut.result()
                results[i] = res
                if on_complete is not None:
                    on_complete(i, res)
    finally:
        if manager is not None:
            manager.shutdown()
    return results
