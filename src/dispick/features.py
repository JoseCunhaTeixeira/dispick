"""What the network reads: the image on the canonical grid, and where each cell stands physically.

Five channels, each the same whatever the survey's units:

- `coherence`: the phase-shift value above the noise floor of random phases, 1/sqrt(N), over
  what is left up to a perfect plane wave (1);
- `column`: each frequency's column over its own maximum, so weak frequencies read as well as
  strong ones;
- `alias`: log2 of the cell's wavelength over twice the receiver spacing: below 0 the array
  aliases (spatial Nyquist);
- `resolution`: log2 of the cell's wavelength over the array's length: above 0 the array
  hardly resolves velocity, and the ridge is broad by nature, not by noise;
- `geometry`: 1 where the array is known, 0 where it is not (then `alias` and `resolution` are
  0 too: the network was trained to pick without them).
"""

from dataclasses import dataclass

import numpy as np

from dispick.grid import CanonicalAxes

CHANNELS: tuple[str, ...] = ("coherence", "column", "alias", "resolution", "geometry")


@dataclass(frozen=True, slots=True)
class Geometry:
    """The receiver array behind an image: its number of receivers and their spacing (m)."""

    n_receivers: int
    spacing: float

    def __post_init__(self) -> None:
        if self.n_receivers < 2:
            raise ValueError(f"an array needs at least 2 receivers, got {self.n_receivers}")
        if not (np.isfinite(self.spacing) and self.spacing > 0):
            raise ValueError(f"the receiver spacing must be > 0, got {self.spacing}")

    @property
    def aperture(self) -> float:
        """The array's length, first to last receiver (m)."""
        return (self.n_receivers - 1) * self.spacing

    @property
    def noise_floor(self) -> float:
        """What N random phases sum to, normalized as the phase shift does: 1/sqrt(N)."""
        return 1.0 / float(np.sqrt(self.n_receivers))

    @classmethod
    def from_positions(cls, positions: np.ndarray) -> Geometry:
        """The geometry of receivers at `positions` (m, along the line), spacing their median
        step."""
        positions = np.sort(np.asarray(positions, dtype=np.float64))
        if positions.size < 2:
            raise ValueError("an array needs at least 2 receivers")
        return cls(n_receivers=int(positions.size), spacing=float(np.median(np.diff(positions))))


def geometry_from_receivers(x: np.ndarray, z: np.ndarray | None = None) -> Geometry | None:
    """The geometry of receivers at positions `x` (and elevations `z`) along a line: their
    count, and the median distance between neighbours along the ground. None when the
    positions are unknown or all equal. Positions, not source offsets: a source inside the
    spread folds the offsets onto each other."""
    x = np.asarray(x, dtype=np.float64)
    z = np.zeros_like(x) if z is None else np.asarray(z, dtype=np.float64)
    if x.size < 2 or not (np.isfinite(x).all() and np.isfinite(z).all()):
        return None
    order = np.argsort(x, kind="stable")
    steps = np.hypot(np.diff(x[order]), np.diff(z[order]))
    steps = steps[steps > 1e-9]
    if steps.size == 0:
        return None
    return Geometry(n_receivers=int(x.size), spacing=float(np.median(steps)))


def normalize_coherence(image: np.ndarray, floor: float) -> np.ndarray:
    """The image's values above `floor`, over what is left up to 1, clipped to [0, 1]. An image
    outside [0, 1] (not a raw phase-shift image) is first rescaled to it, and `floor` ignored."""
    image = np.asarray(image, dtype=np.float32)
    low, high = float(np.nanmin(image)), float(np.nanmax(image))
    if low < -1e-3 or high > 1.0 + 1e-3:
        image = (image - low) / max(high - low, 1e-12)
        floor = 0.0
    return np.clip((image - floor) / (1.0 - floor), 0.0, 1.0)


def input_channels(
    canonical: np.ndarray, axes: CanonicalAxes, geometry: Geometry | None
) -> np.ndarray:
    """The (5, F, V) float32 input for a canonical image (raw phase-shift values)."""
    canonical = np.nan_to_num(np.asarray(canonical, dtype=np.float32), nan=0.0)
    floor = geometry.noise_floor if geometry is not None else 0.0
    coherence = normalize_coherence(canonical, floor)
    peak = np.max(canonical, axis=1, keepdims=True)
    column = np.clip(canonical / np.maximum(peak, 1e-6), 0.0, 1.0)
    out = np.zeros((len(CHANNELS), *canonical.shape), dtype=np.float32)
    out[0] = coherence
    out[1] = column
    if geometry is not None:
        wavelength = axes.velocities[None, :] / np.maximum(axes.frequencies[:, None], 1e-9)
        wavelength = np.maximum(wavelength, 1e-12)
        out[2] = np.clip(np.log2(wavelength / (2 * geometry.spacing)), -4.0, 4.0) / 4.0
        out[3] = np.clip(np.log2(wavelength / geometry.aperture), -6.0, 6.0) / 6.0
        out[4] = 1.0
    return out
