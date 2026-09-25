"""One BLAS thread per data worker.

Each worker draws one image at a time; numpy's matrix products would otherwise start a thread
per core in every worker, and the workers would fight over the machine.
"""

from threadpoolctl import threadpool_limits

_limits: object | None = None


def single_threaded() -> None:
    """Limit this process's BLAS and OpenMP pools to one thread (once per process)."""
    global _limits
    if _limits is None:
        _limits = threadpool_limits(limits=1)
