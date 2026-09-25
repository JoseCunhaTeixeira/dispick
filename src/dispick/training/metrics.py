"""Validation metrics, accumulated over batches.

Velocity errors are relative to M0's true velocity, on the columns where M0 is pickable. The
headline numbers are what a user meets: `pick_precision` (of the points the model keeps, the
share within 5 % of M0), `recall` (of the pickable points, the share the model keeps and gets
within 5 %), and `mode_confusion` (kept points sitting on a higher mode instead of M0).
"""

from dataclasses import dataclass, field

import numpy as np
import torch

TOLERANCE = 0.05


def refined_bins(probabilities: torch.Tensor, window: int = 3) -> torch.Tensor:
    """Each column's argmax, refined by the mean over +-`window` bins around it: (..., V) ->
    (...,) fractional bins."""
    n_bins = probabilities.shape[-1]
    peak = probabilities.argmax(dim=-1, keepdim=True)
    offsets = torch.arange(-window, window + 1, device=probabilities.device)
    index = (peak + offsets).clamp(0, n_bins - 1)
    local = probabilities.gather(-1, index)
    return (local * index.float()).sum(-1) / local.sum(-1).clamp_min(1e-12)


@dataclass
class Metrics:
    sums: dict[str, float] = field(default_factory=dict[str, float])

    def add(self, key: str, value: float) -> None:
        self.sums[key] = self.sums.get(key, 0.0) + value

    @torch.no_grad()
    def update(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        probabilities = torch.softmax(outputs["logits"][:, 0].float(), dim=-1)  # (B, F, V)
        predicted = refined_bins(probabilities)
        bins = batch["target_bins"].float()
        truth = bins[:, 0]
        v_range = batch["v_range"].float()  # (B, 2)
        n_bins = probabilities.shape[-1]
        step = ((v_range[:, 1] - v_range[:, 0]) / (n_bins - 1))[:, None]

        def velocity(b: torch.Tensor) -> torch.Tensor:
            return v_range[:, :1] + b * step

        true_v = velocity(torch.nan_to_num(truth, nan=0.0))
        error = (velocity(predicted) - true_v).abs() / true_v.abs().clamp_min(1e-6)
        finite = torch.isfinite(truth) & (true_v > 0)
        visible = (batch["presence"] > 0.5) & finite
        kept = (torch.sigmoid(outputs["presence"][:, 0].float()) >= 0.5) & finite

        self.add("columns", float(visible.numel()))
        self.add("visible", float(visible.sum()))
        self.add("error_sum", float(error[visible].sum()))
        for threshold in (0.02, 0.05, 0.10):
            self.add(f"within_{threshold:.2f}", float((error[visible] < threshold).sum()))
        self.add("kept", float(kept.sum()))
        self.add("kept_good", float((kept & (error < TOLERANCE)).sum()))
        self.add("found", float((kept & visible & (error < TOLERANCE)).sum()))
        self.add("presence_tp", float((kept & visible).sum()))
        self.add("presence_fp", float((kept & ~visible).sum()))
        self.add("presence_fn", float((~kept & visible).sum()))

        if bins.shape[1] > 1:
            higher = velocity(torch.nan_to_num(bins[:, 1:], nan=-1e9).transpose(0, 1))
            on_higher = (
                ((velocity(predicted)[None] - higher).abs() / higher.abs().clamp_min(1e-6))
                < TOLERANCE
            ).any(dim=0)
            self.add("kept_on_higher", float((kept & on_higher & (error >= TOLERANCE)).sum()))

        targets = batch["image_targets"].float()
        scores = torch.sigmoid(outputs["image"].float())
        pickable = targets[:, 0] > 0.5
        said = scores[:, 0] >= 0.5
        self.add("images", float(pickable.numel()))
        self.add("pickable_tp", float((said & pickable).sum()))
        self.add("pickable_fp", float((said & ~pickable).sum()))
        self.add("pickable_fn", float((~said & pickable).sum()))
        self.add("pickable_correct", float((said == pickable).sum()))
        self.add("quality_error", float((scores[:, 1] - targets[:, 1]).abs().sum()))
        self.add("share_error", float((scores[:, 2] - targets[:, 2]).abs().sum()))

    def summary(self) -> dict[str, float]:
        s = self.sums

        def ratio(a: float, b: float) -> float:
            return a / b if b > 0 else float("nan")

        def f1(tp: float, fp: float, fn: float) -> float:
            return ratio(2 * tp, 2 * tp + fp + fn)

        out = {
            "mape": ratio(s.get("error_sum", 0.0), s.get("visible", 0.0)),
            "acc_2": ratio(s.get("within_0.02", 0.0), s.get("visible", 0.0)),
            "acc_5": ratio(s.get("within_0.05", 0.0), s.get("visible", 0.0)),
            "acc_10": ratio(s.get("within_0.10", 0.0), s.get("visible", 0.0)),
            "pick_precision": ratio(s.get("kept_good", 0.0), s.get("kept", 0.0)),
            "recall": ratio(s.get("found", 0.0), s.get("visible", 0.0)),
            "presence_f1": f1(
                s.get("presence_tp", 0.0), s.get("presence_fp", 0.0), s.get("presence_fn", 0.0)
            ),
            "mode_confusion": ratio(s.get("kept_on_higher", 0.0), s.get("kept", 0.0)),
            "pickable_accuracy": ratio(s.get("pickable_correct", 0.0), s.get("images", 0.0)),
            "pickable_f1": f1(
                s.get("pickable_tp", 0.0), s.get("pickable_fp", 0.0), s.get("pickable_fn", 0.0)
            ),
            "quality_mae": ratio(s.get("quality_error", 0.0), s.get("images", 0.0)),
            "share_mae": ratio(s.get("share_error", 0.0), s.get("images", 0.0)),
        }
        # One number to keep the best checkpoint by: the picks' precision and recall, and the
        # image verdict.
        parts = [out["pick_precision"], out["recall"], out["pickable_f1"]]
        out["score"] = float(np.nanmean(parts)) if not all(np.isnan(parts)) else float("nan")
        return out
