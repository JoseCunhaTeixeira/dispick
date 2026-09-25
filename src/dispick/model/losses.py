"""What the network is trained to minimize.

- Velocity: cross-entropy of each column's velocity distribution against a narrow Gaussian
  around the true bin (label smoothing: a pick one bin off is nearly as good). Where M0 is in
  the image but not pickable, the column still counts, lightly: the network learns where the
  ridge would continue, and its presence head says not to trust it.
- Presence: binary cross-entropy against the pickable mask.
- Image: binary cross-entropy against the image's labels (soft targets for the two ratios).
"""

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.nn import functional


class LossConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sigma_bins: float = Field(default=1.5, gt=0)
    invisible_weight: float = Field(default=0.2, ge=0)
    higher_mode_weight: float = Field(default=0.5, ge=0)
    presence_weight: float = Field(default=1.0, ge=0)
    image_weight: float = Field(default=0.5, ge=0)


def soft_targets(bins: torch.Tensor, n_bins: int, sigma: float) -> torch.Tensor:
    """(..., n_bins) Gaussian distributions centred on the fractional `bins` (NaN: zeros)."""
    grid = torch.arange(n_bins, device=bins.device, dtype=torch.float32)
    centres = torch.nan_to_num(bins.float(), nan=-1e4)[..., None]
    weights = torch.exp(-0.5 * ((grid - centres) / sigma) ** 2)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def picking_loss(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], config: LossConfig
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["logits"].float()  # (B, K, F, V)
    n_modes, n_bins = logits.shape[1], logits.shape[-1]
    bins = batch["target_bins"][:, :n_modes]  # (B, K, F)
    presence = batch["presence"].float()  # (B, F)
    valid = torch.isfinite(bins)

    log_probs = torch.log_softmax(logits, dim=-1)
    cross_entropy = -(soft_targets(bins, n_bins, config.sigma_bins) * log_probs).sum(dim=-1)
    weight = torch.full_like(cross_entropy, config.higher_mode_weight)
    weight[:, 0] = torch.where(presence > 0.5, 1.0, config.invisible_weight)
    weight = weight * valid
    velocity = (cross_entropy * weight).sum() / weight.sum().clamp_min(1.0)

    presence_loss = functional.binary_cross_entropy_with_logits(
        outputs["presence"][:, 0].float(), presence
    )
    image_loss = functional.binary_cross_entropy_with_logits(
        outputs["image"].float(), batch["image_targets"].float()
    )
    total = velocity + config.presence_weight * presence_loss + config.image_weight * image_loss
    return total, {
        "loss": float(total.detach()),
        "velocity": float(velocity.detach()),
        "presence": float(presence_loss.detach()),
        "image": float(image_loss.detach()),
    }
