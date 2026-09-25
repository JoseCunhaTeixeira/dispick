"""Fixed datasets on disk: canonical shards (training, validation) and native benchmarks.

Online generation gives training endless fresh images; fixed sets make validation and
benchmarks reproducible, and let a cloud machine train from files when its CPUs are scarce.

- A shard holds `TrainingExample`s: images quantized to 8 bits (phase-shift values lie in
  [0, 1]; 1/255 is far below any ridge's contrast), their ranges, geometry and targets.
- A benchmark holds `SyntheticSample`s whole, at their own (native) axes, to run the picker
  end to end the way users do, from the image sigpipe computed.
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import h5py
import numpy as np

from dispick import __version__
from dispick.data.examples import TrainingExample
from dispick.features import Geometry
from dispick.synthesis.sample import ImageLabels, SyntheticSample

SHARD_FORMAT = 1
BENCHMARK_FORMAT = 1


def _dataset(group: h5py.File | h5py.Group, key: str) -> h5py.Dataset:
    obj = group[key]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"expected an HDF5 dataset at {key!r}, got {type(obj).__name__}")
    return obj


def _group(group: h5py.File | h5py.Group, key: str) -> h5py.Group:
    obj = group[key]
    if not isinstance(obj, h5py.Group):
        raise TypeError(f"expected an HDF5 group at {key!r}, got {type(obj).__name__}")
    return obj


def write_shard(
    examples: Iterable[TrainingExample], path: Path, count: int, metadata: dict[str, str]
) -> Path:
    """Write `count` examples to `path`; `metadata` (the configurations) goes in its attrs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with h5py.File(tmp, "w") as file:
        file.attrs["format"] = SHARD_FORMAT
        file.attrs["dispick_version"] = __version__
        for key, value in metadata.items():
            file.attrs[key] = value
        written = 0
        for written, example in enumerate(examples, start=1):
            if written == 1:
                _create_shard(file, example, count)
            i = written - 1
            _dataset(file, "image")[i] = np.round(np.clip(example.image, 0.0, 1.0) * 255)
            _dataset(file, "ranges")[i] = [*example.f_range, *example.v_range]
            _dataset(file, "geometry")[i] = [example.geometry.n_receivers, example.geometry.spacing]
            _dataset(file, "target_bins")[i] = example.target_bins
            _dataset(file, "presence")[i] = example.presence.astype(np.uint8)
            _dataset(file, "image_targets")[i] = example.image_targets
            if written == count:
                break
        if written != count:
            raise ValueError(f"expected {count} examples, got {written}")
    tmp.replace(path)
    return path


def _create_shard(file: h5py.File, example: TrainingExample, count: int) -> None:
    n_f, n_v = example.image.shape
    n_modes = example.target_bins.shape[0]
    file.create_dataset("image", (count, n_f, n_v), dtype=np.uint8, chunks=(1, n_f, n_v))
    file.create_dataset("ranges", (count, 4), dtype=np.float64)
    file.create_dataset("geometry", (count, 2), dtype=np.float64)
    file.create_dataset("target_bins", (count, n_modes, n_f), dtype=np.float32)
    file.create_dataset("presence", (count, n_f), dtype=np.uint8)
    file.create_dataset("image_targets", (count, example.image_targets.size), dtype=np.float32)


class Shard:
    """A shard read lazily, example by example (opened per process: h5py files do not survive
    a fork)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._file: h5py.File | None = None
        with h5py.File(self.path, "r") as file:
            if int(file.attrs.get("format", 0)) != SHARD_FORMAT:
                raise ValueError(f"{self.path}: not a shard of format {SHARD_FORMAT}")
            self.size = int(_dataset(file, "image").shape[0])
            self.attrs = {key: str(value) for key, value in file.attrs.items()}

    def __len__(self) -> int:
        return self.size

    def reopen(self) -> None:
        self._file = None

    def __getitem__(self, index: int) -> TrainingExample:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        file = self._file
        ranges = _dataset(file, "ranges")[index]
        n_receivers, spacing = _dataset(file, "geometry")[index]
        return TrainingExample(
            image=_dataset(file, "image")[index].astype(np.float32) / 255.0,
            f_range=(float(ranges[0]), float(ranges[1])),
            v_range=(float(ranges[2]), float(ranges[3])),
            geometry=Geometry(n_receivers=int(n_receivers), spacing=float(spacing)),
            target_bins=np.asarray(_dataset(file, "target_bins")[index], dtype=np.float32),
            presence=_dataset(file, "presence")[index].astype(np.float32),
            image_targets=np.asarray(_dataset(file, "image_targets")[index], dtype=np.float32),
        )


def write_benchmark(
    samples: Iterable[SyntheticSample], path: Path, metadata: dict[str, str]
) -> Path:
    """Write the samples whole, one group each (images in float16)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    count = 0
    with h5py.File(tmp, "w") as file:
        file.attrs["format"] = BENCHMARK_FORMAT
        file.attrs["dispick_version"] = __version__
        for key, value in metadata.items():
            file.attrs[key] = value
        for count, sample in enumerate(samples, start=1):
            group = file.create_group(f"{count - 1:06d}")
            group.create_dataset(
                "fv_map", data=sample.fv_map.astype(np.float16), compression="gzip"
            )
            for key in (
                "frequencies",
                "velocities",
                "offsets",
                "curves",
                "dense_frequencies",
                "dense_curves",
            ):
                group.create_dataset(key, data=getattr(sample, key))
            group.create_dataset("visible", data=sample.visible.astype(np.uint8))
            group.attrs["geometry"] = json.dumps(
                [sample.geometry.n_receivers, sample.geometry.spacing]
            )
            group.attrs["labels"] = json.dumps(
                [sample.labels.pickable, sample.labels.quality, sample.labels.higher_mode_share]
            )
            group.attrs["info"] = json.dumps(sample.info)
        file.attrs["count"] = count
    tmp.replace(path)
    return path


def read_benchmark(path: Path) -> Iterator[SyntheticSample]:
    with h5py.File(path, "r") as file:
        if int(file.attrs.get("format", 0)) != BENCHMARK_FORMAT:
            raise ValueError(f"{path}: not a benchmark of format {BENCHMARK_FORMAT}")
        for name in sorted(file.keys()):
            group = _group(file, name)
            n_receivers, spacing = json.loads(str(group.attrs["geometry"]))
            pickable, quality, share = json.loads(str(group.attrs["labels"]))
            yield SyntheticSample(
                fv_map=np.asarray(_dataset(group, "fv_map")[:], dtype=np.float32),
                frequencies=np.asarray(_dataset(group, "frequencies")[:]),
                velocities=np.asarray(_dataset(group, "velocities")[:]),
                offsets=np.asarray(_dataset(group, "offsets")[:]),
                geometry=Geometry(n_receivers=int(n_receivers), spacing=float(spacing)),
                curves=np.asarray(_dataset(group, "curves")[:]),
                dense_frequencies=np.asarray(_dataset(group, "dense_frequencies")[:]),
                dense_curves=np.asarray(_dataset(group, "dense_curves")[:]),
                visible=np.asarray(_dataset(group, "visible")[:]).astype(bool),
                labels=ImageLabels(bool(pickable), float(quality), float(share)),
                info=json.loads(str(group.attrs["info"])),
            )
