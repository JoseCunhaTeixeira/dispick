"""The picking network: a U-Net over the canonical image, with three heads.

- `logits` (B, K, F, V): for each mode and frequency, a distribution over the velocity bins
  (softmax along V): where the mode's velocity lies;
- `presence` (B, K, F): whether the mode can be picked at that frequency (a logit);
- `image` (B, 3): whether the image is worth picking at all, its quality and how much higher
  modes dominate it (logits of `IMAGE_TARGETS`).

The encoder ends in a few transformer layers over the coarsest feature map: which ridge is M0
is a global question (the lowest coherent branch, continuous over the band, below its
siblings), which convolutions alone answer only locally. Positions enter through a depthwise
convolution, not a fixed table, so the network runs on any grid shape divisible by its total
downsampling.
"""

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.nn import functional

from dispick.features import CHANNELS
from dispick.synthesis.sample import IMAGE_TARGETS


class NetworkConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    in_channels: int = Field(default=len(CHANNELS), ge=1)
    widths: tuple[int, ...] = (32, 64, 128, 256, 384)
    blocks: int = Field(default=2, ge=1)  # residual blocks per stage
    attention_layers: int = Field(default=2, ge=0)
    attention_heads: int = Field(default=8, ge=1)
    n_modes: int = Field(default=1, ge=1)  # modes picked, from M0 up
    n_image_targets: int = Field(default=len(IMAGE_TARGETS), ge=1)
    groups: int = Field(default=8, ge=1)  # group normalization

    @property
    def downsampling(self) -> int:
        return 2 ** (len(self.widths) - 1)


def _norm(channels: int, groups: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(groups, channels), channels)


class ResidualBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, groups: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _norm(c_in, groups),
            nn.SiLU(),
            nn.Conv2d(c_in, c_out, 3, padding=1),
            _norm(c_out, groups),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
        )
        self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.body(x)


class Stage(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, blocks: int, groups: int) -> None:
        super().__init__(
            *[ResidualBlock(c_in if i == 0 else c_out, c_out, groups) for i in range(blocks)]
        )


class GlobalContext(nn.Module):
    """Transformer layers over the coarsest feature map, positions from a depthwise conv."""

    def __init__(self, channels: int, layers: int, heads: int) -> None:
        super().__init__()
        self.position = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=2 * channels,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.position(x)
        b, c, h, w = x.shape
        tokens = self.encoder(x.flatten(2).transpose(1, 2))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class DispersionNet(nn.Module):
    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or NetworkConfig()
        widths, groups = config.widths, config.groups
        self.stem = nn.Conv2d(config.in_channels, widths[0], 3, padding=1)
        self.encoder = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, width in enumerate(widths):
            self.encoder.append(Stage(widths[max(i - 1, 0)], width, config.blocks, groups))
            if i < len(widths) - 1:
                self.downs.append(nn.Conv2d(width, width, 3, stride=2, padding=1))
        self.context = (
            GlobalContext(widths[-1], config.attention_layers, config.attention_heads)
            if config.attention_layers
            else nn.Identity()
        )
        self.ups = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(len(widths) - 2, -1, -1):
            self.ups.append(nn.Conv2d(widths[i + 1], widths[i], 3, padding=1))
            self.decoder.append(Stage(2 * widths[i], widths[i], config.blocks, groups))
        top = widths[0]
        self.velocity_head = nn.Sequential(
            _norm(top, groups), nn.SiLU(), nn.Conv2d(top, config.n_modes, 1)
        )
        # Presence reads each frequency's column of features: an attention-weighted and a
        # max pooling along velocity, then a 1D convolution along frequency.
        self.presence_attention = nn.Conv2d(top, config.n_modes, 1)
        self.presence_head = nn.Sequential(
            nn.Conv1d(2 * top * config.n_modes, 64, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(64, config.n_modes, 1),
        )
        bottom = widths[-1]
        self.image_head = nn.Sequential(
            nn.Linear(2 * bottom, 128), nn.SiLU(), nn.Linear(128, config.n_image_targets)
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        factor = self.config.downsampling
        if x.shape[-2] % factor or x.shape[-1] % factor:
            raise ValueError(
                f"the grid {tuple(x.shape[-2:])} must be divisible by {factor} along both axes"
            )
        x = self.stem(x)
        skips: list[torch.Tensor] = []
        for i, stage in enumerate(self.encoder):
            x = stage(x)
            if i < len(self.downs):
                skips.append(x)
                x = self.downs[i](x)
        x = self.context(x)
        pooled = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        image = self.image_head(pooled)
        for up, stage in zip(self.ups, self.decoder, strict=True):
            skip = skips.pop()
            x = up(functional.interpolate(x, size=skip.shape[-2:], mode="nearest"))
            x = stage(torch.cat([x, skip], dim=1))
        logits = self.velocity_head(x)
        weights = torch.softmax(self.presence_attention(x), dim=-1)  # (B, K, F, V)
        attended = torch.einsum("bkfv,bcfv->bkcf", weights, x).flatten(1, 2)
        peak = x.amax(dim=-1).repeat(1, self.config.n_modes, 1)
        presence = self.presence_head(torch.cat([attended, peak], dim=1))
        return {"logits": logits, "presence": presence, "image": image}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
