"""One synthetic image with its truth, as the generator makes it."""

from dataclasses import dataclass, field

import numpy as np

from dispick.features import Geometry

IMAGE_TARGETS: tuple[str, ...] = ("pickable", "quality", "higher_mode_share")


@dataclass(frozen=True, slots=True)
class ImageLabels:
    pickable: bool
    quality: float
    higher_mode_share: float

    def as_array(self) -> np.ndarray:
        """In the order of `IMAGE_TARGETS`."""
        return np.array(
            [float(self.pickable), self.quality, self.higher_mode_share], dtype=np.float32
        )


@dataclass(frozen=True, slots=True)
class SyntheticSample:
    """A phase-shift image as sigpipe would compute it, and what is true about it.

    `curves` are the modes' velocities at the image's frequencies, and `dense_curves` the same
    on `dense_frequencies` (log-spaced over the image's band), to evaluate them anywhere: both
    are what the phase shift reads (the ground's average along the array when it varies), NaN
    where a mode is absent. `visible` marks where M0 is pickable."""

    fv_map: np.ndarray
    frequencies: np.ndarray
    velocities: np.ndarray
    offsets: np.ndarray
    geometry: Geometry
    curves: np.ndarray
    dense_frequencies: np.ndarray
    dense_curves: np.ndarray
    visible: np.ndarray
    labels: ImageLabels
    info: dict[str, str | float | int] = field(default_factory=dict[str, str | float | int])

    def curves_at(self, frequencies: np.ndarray) -> np.ndarray:
        """The modes' velocities (n_modes, n) at any frequencies within the image's band,
        interpolated in log-frequency; NaN outside the band, at 0 Hz and where a mode is absent."""
        frequencies = np.asarray(frequencies, dtype=np.float64)
        out = np.full((self.dense_curves.shape[0], frequencies.size), np.nan)
        inside = (frequencies >= self.dense_frequencies[0] * (1 - 1e-9)) & (
            frequencies <= self.dense_frequencies[-1] * (1 + 1e-9)
        )
        if not inside.any():
            return out
        log_f = np.log(np.clip(frequencies[inside], self.dense_frequencies[0], None))
        log_grid = np.log(self.dense_frequencies)
        for mode, row in enumerate(self.dense_curves):
            finite = np.isfinite(row)
            if finite.sum() < 2:
                continue
            values = np.interp(log_f, log_grid[finite], row[finite], left=np.nan, right=np.nan)
            out[mode, inside] = values
        return out
