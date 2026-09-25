"""A bank of normalized modal curves: the expensive physics, computed once and reused.

Each entry is one random earth model with the phase velocities and relative excitation of its
Rayleigh modes, on a grid of normalized frequencies f_hat = f H / V shared by all entries (H the
depth to the half-space, V the half-space's Vs). Dispersion is scale-invariant, so one entry
serves any velocity and length scale: c(f) = s_v V c_hat(f s_l H / (s_v V)). The synthesis
draws a scale per image, fitted to the array it simulates.
"""

import json
import logging
import multiprocessing as mp
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dispick import __version__
from dispick.physics.earth import FAMILIES, EarthPrior, Family, LayeredModel, sample_model

logger = logging.getLogger(__name__)

BANK_FORMAT = 1


class BankConfig(BaseModel):
    """What a bank holds, and how it was drawn."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_models: int = Field(default=50_000, gt=0)
    n_modes: int = Field(default=6, ge=1, le=12)
    f_hat_min: float = Field(default=0.02, gt=0)
    f_hat_max: float = Field(default=50.0, gt=0)
    n_frequencies: int = Field(default=384, ge=16)
    excitation_samples: int = Field(default=24, ge=2)
    max_attempts: int = Field(default=20, ge=1)  # models drawn per entry before giving up
    seed: int = 0
    prior: EarthPrior = Field(default_factory=EarthPrior)

    def f_hat(self) -> np.ndarray:
        return np.geomspace(self.f_hat_min, self.f_hat_max, self.n_frequencies)


@dataclass(frozen=True, slots=True)
class BankEntry:
    """One model of the bank, with its normalized curves."""

    index: int
    model: LayeredModel
    velocity: float  # V: the half-space's Vs, m/s
    length: float  # H: the depth to the half-space, m
    f_hat: np.ndarray  # (n_f,)
    c_hat: np.ndarray  # (n_modes, n_f), c / V, NaN below a mode's cut-off
    a_hat: np.ndarray  # (n_modes, n_f), excitation over the strongest mode's, NaN where absent

    @property
    def n_modes(self) -> int:
        return int(self.c_hat.shape[0])

    def curves(
        self,
        frequencies: np.ndarray,
        velocity_scale: float = 1.0,
        length_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Phase velocities (m/s) and relative excitation, both (n_modes, n), at `frequencies`
        (Hz, > 0) of the model with its velocities times `velocity_scale` and its thicknesses
        times `length_scale`. Beyond the bank's grid the curves are held flat (their
        asymptotes); below it, only M0 exists."""
        frequencies = np.asarray(frequencies, dtype=np.float64)
        velocity = self.velocity * velocity_scale
        f_hat = frequencies * (self.length * length_scale) / velocity
        log_grid = np.log(self.f_hat)
        log_f = np.log(np.maximum(f_hat, 1e-300))
        c = np.full((self.n_modes, frequencies.size), np.nan)
        a = np.full((self.n_modes, frequencies.size), np.nan)
        for mode in range(self.n_modes):
            finite = np.isfinite(self.c_hat[mode])
            if not finite.any():
                continue
            grid = log_grid[finite]
            inside = log_f >= grid[0] if mode > 0 else np.ones_like(log_f, dtype=bool)
            c[mode, inside] = np.interp(log_f[inside], grid, self.c_hat[mode, finite]) * velocity
            amplitude = self.a_hat[mode, finite]
            known = np.isfinite(amplitude)
            if known.any():
                a[mode, inside] = np.interp(log_f[inside], grid[known], amplitude[known])
        return c, a


def compute_entry(config: BankConfig, index: int) -> tuple[BankEntry, int]:
    """Entry `index` of the bank `config` describes, and how many models were drawn for it:
    deterministic, whichever process computes it."""
    # Imported here: disba compiles on first use, and the bank's readers do not need it.
    from dispick.physics.dispersion import model_curves
    from dispick.physics.excitation import vertical_excitation

    rng = np.random.default_rng([config.seed, index])
    f_hat = config.f_hat()
    for attempt in range(1, config.max_attempts + 1):
        model = sample_model(config.prior, rng)
        result = model_curves(model, f_hat, config.n_modes)
        if result is None:
            continue
        normalized, c_hat, dc = result
        a_hat = vertical_excitation(
            normalized, f_hat, c_hat, n_samples=config.excitation_samples, dc=dc
        )
        if not np.isfinite(a_hat[0]).any():
            continue
        return (
            BankEntry(
                index=index,
                model=model,
                velocity=normalized.velocity,
                length=normalized.length,
                f_hat=f_hat,
                c_hat=c_hat.astype(np.float32),
                a_hat=a_hat.astype(np.float32),
            ),
            attempt,
        )
    raise RuntimeError(
        f"bank entry {index}: no usable model in {config.max_attempts} draws; widen the prior"
    )


def _compute(task: tuple[BankConfig, int]) -> tuple[BankEntry, int]:
    return compute_entry(*task)


def build_bank(config: BankConfig, path: Path, workers: int | None = None, chunk: int = 64) -> Path:
    """Compute the bank and write it to `path` (HDF5). Entries are computed in parallel and
    written in order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (mp.cpu_count() or 2) - 1)
    max_layers = _max_layers(config)
    tmp = path.with_suffix(path.suffix + ".partial")
    start = time.perf_counter()
    attempts = np.zeros(config.n_models, dtype=np.int16)
    with h5py.File(tmp, "w") as file:
        _create_datasets(file, config, max_layers)
        tasks = ((config, index) for index in range(config.n_models))
        with mp.get_context("spawn").Pool(workers) as pool:
            for done, (entry, attempt) in enumerate(pool.imap(_compute, tasks, chunksize=chunk)):
                _write_entry(file, entry, max_layers)
                attempts[entry.index] = attempt
                if (done + 1) % 1000 == 0 or done + 1 == config.n_models:
                    elapsed = time.perf_counter() - start
                    logger.info(
                        "bank: %d / %d models, %.1f ms each",
                        done + 1,
                        config.n_models,
                        1000 * elapsed / (done + 1) * workers,
                    )
        file.create_dataset("attempts", data=attempts)
    tmp.replace(path)
    return path


def _max_layers(config: BankConfig) -> int:
    # Gradient models reach 16 layers and the half-space, the water table adds one.
    return max(config.prior.n_layers[1] + 3, 16 + 3)


def _create_datasets(file: h5py.File, config: BankConfig, max_layers: int) -> None:
    n, m, k = config.n_models, config.n_modes, config.n_frequencies
    file.attrs["format"] = BANK_FORMAT
    file.attrs["dispick_version"] = __version__
    file.attrs["config"] = config.model_dump_json()
    file.attrs["families"] = json.dumps([str(family) for family in FAMILIES])
    file.create_dataset("f_hat", data=config.f_hat())
    chunks = (1, m, k)
    file.create_dataset("c_hat", shape=(n, m, k), dtype=np.float32, chunks=chunks)
    file.create_dataset("a_hat", shape=(n, m, k), dtype=np.float16, chunks=chunks)
    file.create_dataset("velocity", shape=(n,), dtype=np.float64)
    file.create_dataset("length", shape=(n,), dtype=np.float64)
    file.create_dataset("family", shape=(n,), dtype=np.int8)
    layers = file.create_group("layers")
    layers.create_dataset("count", shape=(n,), dtype=np.int16)
    for name in ("thickness", "vs", "vp", "rho"):
        layers.create_dataset(name, shape=(n, max_layers), dtype=np.float32, fillvalue=np.nan)


def _write_entry(file: h5py.File, entry: BankEntry, max_layers: int) -> None:
    i = entry.index
    _dataset(file, "c_hat")[i] = entry.c_hat
    _dataset(file, "a_hat")[i] = entry.a_hat.astype(np.float16)
    _dataset(file, "velocity")[i] = entry.velocity
    _dataset(file, "length")[i] = entry.length
    _dataset(file, "family")[i] = FAMILIES.index(entry.model.family)
    count = entry.model.n_layers
    if count > max_layers:
        raise ValueError(f"entry {i} has {count} layers, more than the bank's {max_layers}")
    _dataset(file, "layers/count")[i] = count
    for name in ("thickness", "vs", "vp", "rho"):
        row = np.full(max_layers, np.nan, dtype=np.float32)
        row[:count] = getattr(entry.model, name)
        _dataset(file, f"layers/{name}")[i] = row


def _dataset(file: h5py.File | h5py.Group, key: str) -> h5py.Dataset:
    obj = file[key]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"expected an HDF5 dataset at {key!r}, got {type(obj).__name__}")
    return obj


class ModalBank:
    """A bank on disk, read lazily (one entry per read) or loaded whole with `in_memory`.

    Opened per process: an h5py file cannot cross a fork, so each data-loading worker opens
    its own (see `reopen`)."""

    def __init__(self, path: Path, in_memory: bool = False) -> None:
        self.path = Path(path)
        self.in_memory = in_memory
        self._file: h5py.File | None = None
        self._arrays: dict[str, np.ndarray] | None = None
        with h5py.File(self.path, "r") as file:
            if int(file.attrs.get("format", 0)) != BANK_FORMAT:
                raise ValueError(f"{self.path}: not a bank of format {BANK_FORMAT}")
            self.config = BankConfig.model_validate_json(str(file.attrs["config"]))
            self.f_hat = np.asarray(_dataset(file, "f_hat")[:], dtype=np.float64)
            self.families = [Family(name) for name in json.loads(str(file.attrs["families"]))]
            self.size = int(_dataset(file, "velocity").shape[0])
            if in_memory:
                self._arrays = {
                    key: np.asarray(_dataset(file, key)[:])
                    for key in (
                        "c_hat",
                        "a_hat",
                        "velocity",
                        "length",
                        "family",
                        "layers/count",
                        "layers/thickness",
                        "layers/vs",
                        "layers/vp",
                        "layers/rho",
                    )
                }

    def __len__(self) -> int:
        return self.size

    def reopen(self) -> None:
        """Drop the open file handle: the next read opens a new one (call after a fork)."""
        self._file = None

    def _read(self, key: str, index: int) -> np.ndarray:
        if self._arrays is not None:
            return np.asarray(self._arrays[key][index])
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return np.asarray(_dataset(self._file, key)[index])

    def __getitem__(self, index: int) -> BankEntry:
        if not 0 <= index < self.size:
            raise IndexError(f"bank entry {index} out of range 0..{self.size - 1}")
        count = int(self._read("layers/count", index))
        model = LayeredModel(
            thickness=self._read("layers/thickness", index)[:count],
            vs=self._read("layers/vs", index)[:count],
            vp=self._read("layers/vp", index)[:count],
            rho=self._read("layers/rho", index)[:count],
            family=self.families[int(self._read("family", index))],
        )
        return BankEntry(
            index=index,
            model=model,
            velocity=float(self._read("velocity", index)),
            length=float(self._read("length", index)),
            f_hat=self.f_hat,
            c_hat=self._read("c_hat", index).astype(np.float64),
            a_hat=self._read("a_hat", index).astype(np.float64),
        )

    def __iter__(self) -> Iterator[BankEntry]:
        for index in range(self.size):
            yield self[index]
