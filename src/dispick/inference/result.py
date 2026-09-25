"""What the picker returns: a curve with per-point diagnostics, and a verdict on the image."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

type Verdict = Literal["pickable", "doubtful", "unpickable"]


@dataclass(frozen=True, slots=True)
class ImageAssessment:
    """The network's view of the whole image.

    - `pickable`: probability that some stretch of M0 can be picked, spanning at least half
      an octave of wavelength;
    - `quality`: the expected wavelength span of that stretch, over three octaves (capped at
      1): 0.33 is one octave, a curve twice as long in wavelength as it is short;
    - `higher_mode_share`: expected share of M0's frequencies where a higher mode is the
      brightest ridge: the image invites mode jumps.
    """

    pickable: float
    quality: float
    higher_mode_share: float
    thresholds: tuple[float, float] = (0.3, 0.7)

    @property
    def verdict(self) -> Verdict:
        low, high = self.thresholds
        if self.pickable >= high:
            return "pickable"
        if self.pickable < low:
            return "unpickable"
        return "doubtful"

    def to_dict(self) -> dict[str, float | str]:
        return {
            "verdict": self.verdict,
            "pickable": round(self.pickable, 4),
            "quality": round(self.quality, 4),
            "higher_mode_share": round(self.higher_mode_share, 4),
        }


@dataclass(frozen=True, slots=True)
class PickResult:
    """M0 at each of the image's own frequencies.

    `velocities` and `uncertainties` (the spread of the network's estimate, m/s) are given at
    every frequency where the image spans M0's estimate; `presence` is the probability that M0
    is pickable there, and `picked` the points kept: presence at least `threshold`, within the
    widest continuous stretch. The curve for an inversion is the picked points."""

    frequencies: np.ndarray
    velocities: np.ndarray
    uncertainties: np.ndarray
    presence: np.ndarray
    picked: np.ndarray
    image: ImageAssessment
    threshold: float
    mode: int = 0

    @property
    def n_picked(self) -> int:
        return int(np.sum(self.picked))

    @property
    def curve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(frequencies, velocities, uncertainties) of the picked points."""
        keep = self.picked
        return self.frequencies[keep], self.velocities[keep], self.uncertainties[keep]

    @property
    def wavelength_range(self) -> tuple[float, float] | None:
        """Shortest and longest picked wavelengths (m), or None when nothing is picked."""
        f, v, _ = self.curve
        if f.size == 0:
            return None
        wavelengths = v / f
        return float(np.min(wavelengths)), float(np.max(wavelengths))

    def to_dict(self) -> dict[str, object]:
        f, v, e = self.curve
        return {
            "mode": self.mode,
            "image": self.image.to_dict(),
            "threshold": self.threshold,
            "n_picked": self.n_picked,
            "wavelength_range_m": self.wavelength_range,
            "curve": {
                "frequency_hz": [round(float(x), 6) for x in f],
                "velocity_m_s": [round(float(x), 3) for x in v],
                "uncertainty_m_s": [round(float(x), 3) for x in e],
                "presence": [round(float(x), 4) for x in self.presence[self.picked]],
            },
        }
