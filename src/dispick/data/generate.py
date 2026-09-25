"""Fixed datasets drawn in parallel: shards of training examples, and native benchmarks.

Sample i of a dataset seeded s is drawn from its own stream [s, i], whichever process draws it,
so a dataset is the same however many workers make it.
"""

import multiprocessing as mp
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dispick.data.examples import TrainingExample, make_example
from dispick.data.shards import write_benchmark, write_shard
from dispick.data.threads import single_threaded
from dispick.grid import CanonicalGrid
from dispick.physics.bank import ModalBank
from dispick.synthesis.config import SynthesisConfig
from dispick.synthesis.generator import SyntheticGenerator
from dispick.synthesis.sample import SyntheticSample


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    bank: Path
    synthesis: SynthesisConfig
    seed: int
    grid: tuple[int, int] = (256, 256)
    n_modes: int = 4


_generator: SyntheticGenerator | None = None
_spec: DatasetSpec | None = None


def _init(spec: DatasetSpec) -> None:
    global _generator, _spec
    single_threaded()
    _spec = spec
    _generator = SyntheticGenerator(ModalBank(spec.bank), spec.synthesis)


def draw_sample(index: int) -> SyntheticSample:
    """Sample `index` of the dataset this process was initialized for."""
    if _generator is None or _spec is None:
        raise RuntimeError("the process was not initialized with a dataset spec")
    return _generator.sample(np.random.default_rng([_spec.seed, index]))


def draw_example(index: int) -> TrainingExample:
    if _spec is None:
        raise RuntimeError("the process was not initialized with a dataset spec")
    return make_example(draw_sample(index), CanonicalGrid(*_spec.grid), _spec.n_modes)


def _parallel[T](
    spec: DatasetSpec, count: int, workers: int, function: Callable[[int], T]
) -> Iterator[T]:
    if workers <= 1:
        _init(spec)
        for index in range(count):
            yield function(index)
        return
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(spec,)) as pool:
        yield from pool.imap(function, range(count), chunksize=16)


def _metadata(spec: DatasetSpec) -> dict[str, str]:
    return {
        "synthesis": spec.synthesis.model_dump_json(),
        "seed": str(spec.seed),
        "bank": str(spec.bank),
        "grid": f"{spec.grid[0]}x{spec.grid[1]}",
    }


def build_shard(spec: DatasetSpec, path: Path, count: int, workers: int = 1) -> Path:
    examples = _parallel(spec, count, workers, draw_example)
    return write_shard(examples, path, count, _metadata(spec))


def build_benchmark(spec: DatasetSpec, path: Path, count: int, workers: int = 1) -> Path:
    samples = _parallel(spec, count, workers, draw_sample)
    return write_benchmark(samples, path, _metadata(spec))
