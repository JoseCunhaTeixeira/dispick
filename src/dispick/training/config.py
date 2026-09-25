"""A training run's configuration, read from YAML and saved with every checkpoint."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dispick.model.losses import LossConfig
from dispick.model.network import NetworkConfig
from dispick.synthesis.config import SynthesisConfig


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DataConfig(_Section):
    bank: Path  # the modal bank (`dispick bank build`)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    grid: tuple[int, int] = (256, 256)  # canonical frequencies x velocities
    stored_modes: int = Field(default=4, ge=1)  # modes in the targets: M0 picked, the rest judged
    source: Literal["online", "shards"] = "online"
    train_shards: tuple[Path, ...] = ()
    validation: Path | None = None  # a shard (`dispick data shard`)
    geometry_dropout: float = Field(default=0.1, ge=0, le=1)
    batch_size: int = Field(default=32, ge=1)  # per process
    workers: int = Field(default=8, ge=0)
    prefetch_factor: int = Field(default=4, ge=1)
    bank_in_memory: bool = False

    @model_validator(mode="after")
    def _check(self) -> DataConfig:
        if self.source == "shards" and not self.train_shards:
            raise ValueError("source 'shards' needs train_shards")
        return self


class OptimConfig(_Section):
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.05, ge=0)
    betas: tuple[float, float] = (0.9, 0.999)
    steps: int = Field(default=100_000, gt=0)
    warmup_steps: int = Field(default=2_000, ge=0)
    min_lr_ratio: float = Field(default=0.02, ge=0, le=1)
    grad_clip: float = Field(default=1.0, gt=0)
    ema_decay: float = Field(default=0.999, ge=0, lt=1)


class RuntimeConfig(_Section):
    device: str = "auto"  # "auto": the GPU (CUDA or ROCm) when there is one
    precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    compile: bool = False
    threads: int | None = Field(default=None, ge=1)  # PyTorch's CPU threads; None: its default
    seed: int = 0
    log_every: int = Field(default=50, ge=1)
    validate_every: int = Field(default=2_000, ge=1)
    checkpoint_every: int = Field(default=2_000, ge=1)
    validation_batches: int | None = Field(default=None, ge=1)  # None: the whole shard


class TrainConfig(_Section):
    output: Path  # the run's folder: checkpoints, logs, the configuration
    data: DataConfig
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def _check(self) -> TrainConfig:
        factor = self.network.downsampling
        if any(size % factor for size in self.data.grid):
            raise ValueError(f"grid {self.data.grid} must be divisible by {factor}")
        if self.network.n_modes > self.data.stored_modes:
            raise ValueError("the network cannot pick more modes than the targets store")
        return self

    @classmethod
    def from_yaml(cls, path: Path, overrides: dict[str, object] | None = None) -> TrainConfig:
        """Read `path`; relative paths in it are taken from its folder. `overrides` maps
        dotted keys (runtime.device, optim.steps, ...) to values."""
        path = Path(path)
        raw = yaml.safe_load(path.read_text()) or {}
        for dotted, value in (overrides or {}).items():
            node = raw
            *parents, leaf = dotted.split(".")
            for key in parents:
                node = node.setdefault(key, {})
            node[leaf] = value
        config = cls.model_validate(raw)
        return config.resolved(path.parent)

    def resolved(self, base: Path) -> TrainConfig:
        def resolve(p: Path) -> Path:
            return p if p.is_absolute() else (base / p).resolve()

        data = self.data.model_copy(
            update={
                "bank": resolve(self.data.bank),
                "train_shards": tuple(resolve(p) for p in self.data.train_shards),
                "validation": resolve(self.data.validation) if self.data.validation else None,
            }
        )
        return self.model_copy(update={"output": resolve(self.output), "data": data})

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)
