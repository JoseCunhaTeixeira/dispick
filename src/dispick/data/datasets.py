"""PyTorch datasets: endless synthetic images, or a fixed shard."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from dispick.data.examples import TrainingExample, make_example
from dispick.data.shards import Shard
from dispick.data.threads import single_threaded
from dispick.grid import CanonicalGrid
from dispick.physics.bank import ModalBank
from dispick.synthesis.config import SynthesisConfig
from dispick.synthesis.generator import SyntheticGenerator

type Batch = dict[str, torch.Tensor]


def to_tensors(example: TrainingExample, with_geometry: bool = True) -> Batch:
    return {
        "inputs": torch.from_numpy(example.inputs(with_geometry)),
        "target_bins": torch.from_numpy(example.target_bins),
        "presence": torch.from_numpy(example.presence),
        "image_targets": torch.from_numpy(example.image_targets),
        "v_range": torch.tensor(example.v_range, dtype=torch.float64),
    }


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


class OnlineDataset(IterableDataset[Batch]):
    """Fresh synthetic images forever. Each (seed, restart, rank, worker) draws its own
    stream, so processes never repeat each other and a resumed run does not replay the
    images it already saw.

    The rank is read when the dataset is made, in the training process: data workers start
    through forkserver (Python 3.14's default), fresh processes where torch.distributed is not
    initialized, so they would all read rank 0 and every GPU would train on the same images."""

    def __init__(
        self,
        bank_path: Path,
        synthesis: SynthesisConfig,
        grid: CanonicalGrid,
        n_modes: int,
        seed: int,
        geometry_dropout: float = 0.1,
        restart: int = 0,
        bank_in_memory: bool = False,
        rank: int | None = None,
    ) -> None:
        self.rank = _rank() if rank is None else rank
        self.bank_path = Path(bank_path)
        self.synthesis = synthesis
        self.grid = grid
        self.n_modes = n_modes
        self.seed = seed
        self.geometry_dropout = geometry_dropout
        self.restart = restart
        self.bank_in_memory = bank_in_memory

    def __iter__(self):  # noqa: ANN204 -- torch's own signature
        single_threaded()
        info = get_worker_info()
        worker = info.id if info is not None else 0
        rng = np.random.default_rng([self.seed, self.restart, self.rank, worker])
        generator = SyntheticGenerator(
            ModalBank(self.bank_path, in_memory=self.bank_in_memory), self.synthesis
        )
        while True:
            example = make_example(generator.sample(rng), self.grid, self.n_modes)
            yield to_tensors(example, with_geometry=rng.random() >= self.geometry_dropout)


class ShardDataset(Dataset[Batch]):
    """A shard's examples; the geometry dropped for a fixed share of them (by index)."""

    def __init__(self, path: Path, geometry_dropout: float = 0.0, seed: int = 0) -> None:
        self.shard = Shard(path)
        drop = np.random.default_rng(seed).random(len(self.shard)) < geometry_dropout
        self.drop = drop

    def __len__(self) -> int:
        return len(self.shard)

    def __getitem__(self, index: int) -> Batch:
        return to_tensors(self.shard[index], with_geometry=not bool(self.drop[index]))


def worker_init(_: int) -> None:
    """Give each data-loading worker its own HDF5 handles, and one BLAS thread."""
    single_threaded()
    info = get_worker_info()
    if info is not None and isinstance(info.dataset, ShardDataset):
        info.dataset.shard.reopen()
