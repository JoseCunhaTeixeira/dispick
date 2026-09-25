"""A trained checkpoint to a deployable model: ONNX weights and their card.

The ONNX model runs with onnxruntime alone, so sigpipe, PAC and PACo pick without PyTorch.
The export checks itself: the ONNX outputs must match PyTorch's on the same input.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dispick.features import CHANNELS
from dispick.inference.backends import ModelCard, OnnxBackend
from dispick.model.network import DispersionNet, NetworkConfig
from dispick.training.trainer import load_checkpoint

logger = logging.getLogger(__name__)

OUTPUTS = ("logits", "presence", "image")


class _Tupled(nn.Module):
    """The network with its outputs as a tuple, in `OUTPUTS` order."""

    def __init__(self, network: DispersionNet) -> None:
        super().__init__()
        self.network = network

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.network(inputs)
        return tuple(outputs[name] for name in OUTPUTS)


def export_onnx(checkpoint: Path, output: Path, opset: int = 18, tolerance: float = 1e-3) -> Path:
    """Write `output` (.onnx) and its card (.json) from the checkpoint's averaged weights."""
    state = load_checkpoint(Path(checkpoint))
    config = state["config"]
    network = DispersionNet(NetworkConfig.model_validate(config["network"]))
    network.load_state_dict(state["ema"])
    network.eval()
    grid = tuple(config["data"]["grid"])
    example = torch.rand(2, len(CHANNELS), *grid)
    output = Path(output).with_suffix(".onnx")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        _Tupled(network),
        (example,),
        str(output),
        input_names=["inputs"],
        output_names=list(OUTPUTS),
        dynamic_axes={name: {0: "batch"} for name in ("inputs", *OUTPUTS)},
        opset_version=opset,
        dynamo=False,
    )
    card = ModelCard.from_checkpoint(state)
    output.with_suffix(".json").write_text(card.to_json())

    with torch.inference_mode():
        expected = network(example)
    got = OnnxBackend(output).run(example.numpy())
    for name in OUTPUTS:
        difference = float(np.max(np.abs(got[name] - expected[name].numpy())))
        if difference > tolerance:
            raise RuntimeError(f"ONNX export differs from PyTorch on {name!r} by {difference:.2e}")
    logger.info("exported %s (step %d) to %s", checkpoint, int(state["step"]), output)
    return output
