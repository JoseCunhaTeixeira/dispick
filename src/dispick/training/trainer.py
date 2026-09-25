"""The training loop.

One process per GPU: run it plainly for one device, or under `torchrun` for several (it reads
RANK, WORLD_SIZE and LOCAL_RANK). Mixed precision (bf16 where the GPU has it), gradient
clipping, a warm-up then cosine learning rate, and an exponential moving average of the weights,
which is what gets validated, kept and exported. Everything needed to resume is in `last.pt`;
the best validation score's weights are in `best.pt`.
"""

import copy
import json
import logging
import math
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dispick import __version__
from dispick.data.datasets import Batch, OnlineDataset, ShardDataset, worker_init
from dispick.grid import CanonicalGrid
from dispick.model.losses import picking_loss
from dispick.model.network import DispersionNet, count_parameters
from dispick.training.config import OptimConfig, RuntimeConfig, TrainConfig
from dispick.training.metrics import Metrics

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT = 1


@dataclass(frozen=True, slots=True)
class Runtime:
    rank: int
    world: int
    local_rank: int
    device: torch.device
    dtype: torch.dtype

    @property
    def main(self) -> bool:
        return self.rank == 0

    def autocast(self) -> Any:  # noqa: ANN401 -- torch's context manager type
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.dtype != torch.float32,
        )


def setup_runtime(config: RuntimeConfig) -> Runtime:
    if config.threads is not None:
        torch.set_num_threads(config.threads)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if config.device == "auto":
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    precision = config.precision
    if precision == "auto":
        gpu_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        precision = "bf16" if gpu_bf16 else "fp32"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    return Runtime(rank=rank, world=world, local_rank=local_rank, device=device, dtype=dtype)


def learning_rate_factor(step: int, config: OptimConfig) -> float:
    """Linear warm-up, then a cosine decay to `min_lr_ratio`."""
    if step < config.warmup_steps:
        return (step + 1) / config.warmup_steps
    progress = min(1.0, (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps))
    return config.min_lr_ratio + (1 - config.min_lr_ratio) * 0.5 * (
        1 + math.cos(math.pi * progress)
    )


class EMA:
    """An exponential moving average of a model's weights (buffers copied as they are)."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        # Early on the average would drag the random initialization: ramp the decay up.
        decay = min(self.decay, (1 + step) / (10 + step))
        for average, current in zip(self.model.parameters(), model.parameters(), strict=True):
            average.lerp_(current.detach(), 1 - decay)
        for average, current in zip(self.model.buffers(), model.buffers(), strict=True):
            average.copy_(current)


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """No weight decay on biases and normalization weights."""
    decayed, plain = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (plain if parameter.ndim <= 1 or name.endswith(".bias") else decayed).append(parameter)
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]


def _share_through_files_if_shm_is_small(minimum: int = 1 << 30) -> None:
    """Data workers hand batches over through /dev/shm; some containers give it 64 MB, not
    enough: share through files instead."""
    try:
        stats = os.statvfs("/dev/shm")
    except OSError:
        return
    if stats.f_bavail * stats.f_frsize < minimum:
        torch.multiprocessing.set_sharing_strategy("file_system")
        logger.warning("/dev/shm is small: data workers share batches through files")


def train_loader(config: TrainConfig, runtime: Runtime, restart: int) -> DataLoader[Batch]:
    data = config.data
    if data.workers:
        _share_through_files_if_shm_is_small()
    grid = CanonicalGrid(*data.grid)
    pin = runtime.device.type == "cuda"
    if data.source == "online":
        dataset = OnlineDataset(
            data.bank,
            data.synthesis,
            grid,
            data.stored_modes,
            seed=config.runtime.seed,
            geometry_dropout=data.geometry_dropout,
            restart=restart,
            bank_in_memory=data.bank_in_memory,
        )
        return DataLoader(
            dataset,
            batch_size=data.batch_size,
            num_workers=data.workers,
            prefetch_factor=data.prefetch_factor if data.workers else None,
            persistent_workers=data.workers > 0,
            pin_memory=pin,
        )
    shards = torch.utils.data.ConcatDataset(
        [
            ShardDataset(path, data.geometry_dropout, config.runtime.seed)
            for path in data.train_shards
        ]
    )
    sampler = DistributedSampler(
        shards, runtime.world, runtime.rank, shuffle=True, seed=config.runtime.seed
    )
    return DataLoader(
        shards,
        batch_size=data.batch_size,
        sampler=sampler,
        num_workers=data.workers,
        prefetch_factor=data.prefetch_factor if data.workers else None,
        persistent_workers=data.workers > 0,
        pin_memory=pin,
        worker_init_fn=worker_init,
        drop_last=True,
    )


def validation_loader(config: TrainConfig) -> DataLoader[Batch] | None:
    if config.data.validation is None:
        return None
    dataset = ShardDataset(config.data.validation)
    grid = "x".join(str(size) for size in config.data.grid)
    if dataset.shard.attrs.get("grid", grid) != grid:
        raise ValueError(
            f"validation shard {config.data.validation} is on a {dataset.shard.attrs['grid']} "
            f"grid, the network on {grid}: rebuild it with --grid {grid.replace('x', ' ')}"
        )
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        num_workers=min(config.data.workers, 4),
        worker_init_fn=worker_init,
        shuffle=False,
    )


def to_device(batch: Batch, device: torch.device) -> Batch:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader[Batch],
    config: TrainConfig,
    runtime: Runtime,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    metrics = Metrics()
    losses: dict[str, float] = {}
    count = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = to_device(batch, runtime.device)
        with runtime.autocast():
            outputs = model(batch["inputs"])
        _, parts = picking_loss(outputs, batch, config.loss)
        for key, value in parts.items():
            losses[key] = losses.get(key, 0.0) + value
        count += 1
        metrics.update(outputs, batch)
    summary = metrics.summary()
    summary.update({f"val_{key}": value / max(count, 1) for key, value in losses.items()})
    return summary


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: not a dispick checkpoint of format {CHECKPOINT_FORMAT}")
    return state


def _batches(loader: DataLoader[Batch], epoch_start: int) -> Iterator[Batch]:
    """The loader's batches forever (a shard loader starts a new epoch when it runs out)."""
    epoch = epoch_start
    while True:
        sampler = getattr(loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def train(config: TrainConfig, resume: bool = True) -> Path:
    """Train as `config` says, resuming from its `last.pt` if there is one; returns the path of
    the checkpoint to use (the best validated, or the last)."""
    runtime = setup_runtime(config.runtime)
    output = config.output
    if runtime.main:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.yaml").write_text(config.to_yaml())
    torch.manual_seed(config.runtime.seed + runtime.rank)

    model = DispersionNet(config.network).to(runtime.device)
    ema = EMA(model, config.optim.ema_decay)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, config.optim.weight_decay),
        lr=config.optim.lr,
        betas=config.optim.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_factor(step, config.optim)
    )
    scaler = torch.amp.GradScaler(enabled=runtime.dtype == torch.float16)
    start, best = 0, -math.inf
    last_path, best_path = output / "last.pt", output / "best.pt"
    if resume and last_path.exists():
        state = load_checkpoint(last_path, runtime.device)
        model.load_state_dict(state["model"])
        ema.model.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, best = int(state["step"]), float(state["best_score"])
        if runtime.main:
            logger.info("resumed from %s at step %d", last_path, start)
    if runtime.main:
        logger.info(
            "training %.2fM parameters on %s (%s), %d process(es)",
            count_parameters(model) / 1e6,
            runtime.device,
            runtime.dtype,
            runtime.world,
        )

    network: nn.Module = model
    if runtime.world > 1:
        network = DistributedDataParallel(
            model, device_ids=[runtime.local_rank] if runtime.device.type == "cuda" else None
        )
    if config.runtime.compile:
        network = torch.compile(network)  # pyright: ignore[reportAssignmentType]

    batches = _batches(train_loader(config, runtime, restart=start), epoch_start=start)
    validation = validation_loader(config) if runtime.main else None
    log = (output / "metrics.jsonl").open("a") if runtime.main else None
    writer = _tensorboard(output) if runtime.main else None

    def checkpoint(step: int) -> dict[str, Any]:
        return {
            "format": CHECKPOINT_FORMAT,
            "dispick_version": __version__,
            "step": step,
            "best_score": best,
            "config": config.model_dump(mode="json"),
            "model": model.state_dict(),
            "ema": ema.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }

    running: dict[str, float] = {}
    tick, seen = time.perf_counter(), 0
    for step in range(start + 1, config.optim.steps + 1):
        network.train()
        batch = to_device(next(batches), runtime.device)
        with runtime.autocast():
            outputs = network(batch["inputs"])
        loss, parts = picking_loss(outputs, batch, config.loss)
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            logger.warning("step %d: non-finite loss %s, skipped", step, float(loss))
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model, step)
        seen += batch["inputs"].shape[0] * runtime.world
        for key, value in parts.items():
            running[key] = running.get(key, 0.0) + value
        running["grad_norm"] = running.get("grad_norm", 0.0) + float(grad_norm)
        running["count"] = running.get("count", 0.0) + 1

        if runtime.main and step % config.runtime.log_every == 0:
            elapsed = time.perf_counter() - tick
            count = running.pop("count")
            record = {key: value / count for key, value in running.items()}
            record |= {
                "step": float(step),
                "lr": float(scheduler.get_last_lr()[0]),
                "images_per_s": seen / elapsed,
            }
            _write(log, writer, "train", record)
            logger.info(
                "step %d  loss %.4f  velocity %.4f  presence %.4f  image %.4f  %.0f img/s",
                step, record["loss"], record["velocity"], record["presence"], record["image"],
                record["images_per_s"],
            )  # fmt: skip
            running, tick, seen = {}, time.perf_counter(), 0

        last_step = step == config.optim.steps
        if validation is not None and (step % config.runtime.validate_every == 0 or last_step):
            summary = validate(
                ema.model, validation, config, runtime, config.runtime.validation_batches
            )
            _write(log, writer, "validation", summary | {"step": float(step)})
            logger.info(
                "validation at %d: score %.4f  precision %.4f  recall %.4f  acc@5%% %.4f  "
                "mode confusion %.4f  pickable F1 %.4f",
                step, summary["score"], summary["pick_precision"], summary["recall"],
                summary["acc_5"], summary["mode_confusion"], summary["pickable_f1"],
            )  # fmt: skip
            if math.isfinite(summary["score"]) and summary["score"] > best:
                best = summary["score"]
                save_checkpoint(best_path, checkpoint(step) | {"metrics": summary})
        if runtime.main and (step % config.runtime.checkpoint_every == 0 or last_step):
            save_checkpoint(last_path, checkpoint(step))
        if runtime.world > 1 and (step % config.runtime.validate_every == 0 or last_step):
            torch.distributed.barrier()

    if log is not None:
        log.close()
    if writer is not None:
        writer.close()
    if runtime.world > 1:
        torch.distributed.destroy_process_group()
    return best_path if best_path.exists() else last_path


def _tensorboard(output: Path) -> Any:  # noqa: ANN401 -- optional dependency
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    return SummaryWriter(log_dir=str(output / "tensorboard"))


def _write(log: Any, writer: Any, split: str, record: dict[str, float]) -> None:  # noqa: ANN401
    if log is not None:
        log.write(json.dumps({"split": split} | record) + "\n")
        log.flush()
    if writer is not None:
        step = int(record["step"])
        for key, value in record.items():
            if key != "step" and isinstance(value, int | float) and math.isfinite(value):
                writer.add_scalar(f"{split}/{key}", value, step)
