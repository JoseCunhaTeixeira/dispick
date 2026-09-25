"""Running the network: ONNX Runtime (no PyTorch needed, the deployed way) or PyTorch.

A model is a file and its card: `model.onnx` with `model.json`, or a training checkpoint
(`.pt`), whose card is built from the configuration it carries.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

CARD_FORMAT = 1


@dataclass(frozen=True, slots=True)
class ModelCard:
    """What a model expects and was trained on."""

    grid: tuple[int, int]
    channels: tuple[str, ...]
    n_modes: int
    image_targets: tuple[str, ...]
    dispick_version: str
    details: dict[str, Any]  # network, labels, synthesis, training metrics

    def to_json(self) -> str:
        return json.dumps(
            {
                "format": CARD_FORMAT,
                "grid": list(self.grid),
                "channels": list(self.channels),
                "n_modes": self.n_modes,
                "image_targets": list(self.image_targets),
                "dispick_version": self.dispick_version,
                "details": self.details,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> ModelCard:
        raw = json.loads(text)
        if raw.get("format") != CARD_FORMAT:
            raise ValueError(f"not a dispick model card of format {CARD_FORMAT}")
        return cls(
            grid=(int(raw["grid"][0]), int(raw["grid"][1])),
            channels=tuple(raw["channels"]),
            n_modes=int(raw["n_modes"]),
            image_targets=tuple(raw["image_targets"]),
            dispick_version=str(raw["dispick_version"]),
            details=dict(raw["details"]),
        )

    @classmethod
    def from_checkpoint(cls, state: dict[str, Any]) -> ModelCard:
        from dispick.features import CHANNELS
        from dispick.synthesis.sample import IMAGE_TARGETS

        config = state["config"]
        return cls(
            grid=(int(config["data"]["grid"][0]), int(config["data"]["grid"][1])),
            channels=CHANNELS,
            n_modes=int(config["network"]["n_modes"]),
            image_targets=IMAGE_TARGETS,
            dispick_version=str(state["dispick_version"]),
            details={
                "network": config["network"],
                "labels": config["data"]["synthesis"]["labels"],
                "synthesis": config["data"]["synthesis"],
                "training": {
                    "step": int(state["step"]),
                    "metrics": state.get("metrics", {}),
                },
            },
        )


class Backend(Protocol):
    def run(self, inputs: np.ndarray) -> dict[str, np.ndarray]:
        """Outputs (logits, presence, image) for a (B, C, F, V) float32 batch."""
        ...


class OnnxBackend:
    def __init__(
        self, path: Path, providers: list[str] | None = None, threads: int | None = None
    ) -> None:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads is not None:
            # Several pickers in parallel processes (PACo's workers): one pool each, not all cores.
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        available = onnxruntime.get_available_providers()
        chosen = [p for p in (providers or ["CPUExecutionProvider"]) if p in available]
        self.session = onnxruntime.InferenceSession(
            str(path), sess_options=options, providers=chosen or ["CPUExecutionProvider"]
        )
        self.outputs = [output.name for output in self.session.get_outputs()]

    def run(self, inputs: np.ndarray) -> dict[str, np.ndarray]:
        values = self.session.run(None, {"inputs": inputs.astype(np.float32)})
        return {name: np.asarray(value) for name, value in zip(self.outputs, values, strict=True)}


class TorchBackend:
    def __init__(self, state: dict[str, Any], device: str = "cpu") -> None:
        import torch

        from dispick.model.network import DispersionNet, NetworkConfig

        self.torch = torch
        self.device = torch.device(device)
        self.model = DispersionNet(NetworkConfig.model_validate(state["config"]["network"]))
        self.model.load_state_dict(state["ema"])
        self.model.to(self.device).eval()

    def run(self, inputs: np.ndarray) -> dict[str, np.ndarray]:
        torch = self.torch
        with torch.inference_mode():
            outputs = self.model(torch.from_numpy(inputs.astype(np.float32)).to(self.device))
        return {key: value.float().cpu().numpy() for key, value in outputs.items()}


def load_backend(
    path: Path, device: str = "cpu", threads: int | None = None
) -> tuple[Backend, ModelCard]:
    """The model at `path` and its card; `threads` caps the CPU threads it uses."""
    path = Path(path)
    if path.suffix == ".onnx":
        card_path = path.with_suffix(".json")
        if not card_path.exists():
            raise FileNotFoundError(f"{path}: its model card {card_path.name} is missing")
        providers = ["CUDAExecutionProvider", "ROCMExecutionProvider"] if device != "cpu" else None
        return OnnxBackend(path, providers, threads), ModelCard.from_json(card_path.read_text())
    if path.suffix == ".pt":
        from dispick.training.trainer import load_checkpoint

        state = load_checkpoint(path)
        return TorchBackend(state, device), ModelCard.from_checkpoint(state)
    raise ValueError(f"{path}: a model is an .onnx file (with its .json card) or a .pt checkpoint")
