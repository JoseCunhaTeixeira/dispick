"""The fixed grid the network sees, whatever the image's own axes.

A dispersion image comes with its own frequency and velocity axes: any range, any step, any
size. The network works on one grid shape: each image is resampled onto linear axes spanning
its own first to last frequency and velocity, with a tent kernel as wide as the coarser of the
two steps (linear interpolation when refining, an average when coarsening, so a fine image is
not aliased). Training and picking go through the same functions.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CanonicalGrid:
    """The network's input shape: frequencies x velocities."""

    n_frequencies: int = 256
    n_velocities: int = 256

    def __post_init__(self) -> None:
        if self.n_frequencies < 8 or self.n_velocities < 8:
            raise ValueError("a canonical grid needs at least 8 x 8 cells")

    def axes(self, f_range: tuple[float, float], v_range: tuple[float, float]) -> CanonicalAxes:
        return CanonicalAxes(
            frequencies=np.linspace(f_range[0], f_range[1], self.n_frequencies),
            velocities=np.linspace(v_range[0], v_range[1], self.n_velocities),
        )


@dataclass(frozen=True, slots=True)
class CanonicalAxes:
    """The physical frequency (Hz) and velocity (m/s) of each canonical row and column."""

    frequencies: np.ndarray
    velocities: np.ndarray

    @property
    def f_range(self) -> tuple[float, float]:
        return float(self.frequencies[0]), float(self.frequencies[-1])

    @property
    def v_range(self) -> tuple[float, float]:
        return float(self.velocities[0]), float(self.velocities[-1])

    def velocity_of_bin(self, bins: np.ndarray) -> np.ndarray:
        """Velocities (m/s) of fractional velocity-bin coordinates."""
        low, high = self.v_range
        return low + np.asarray(bins, dtype=np.float64) * (high - low) / (self.velocities.size - 1)

    def bin_of_velocity(self, velocities: np.ndarray) -> np.ndarray:
        """Fractional velocity-bin coordinates of velocities (m/s)."""
        low, high = self.v_range
        return (
            (np.asarray(velocities, dtype=np.float64) - low)
            * (self.velocities.size - 1)
            / (high - low)
        )


def check_axis(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be a 1D axis of at least 2 values, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def resampling_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Weights (len(target), len(source)) taking values on the increasing axis `source` to the
    increasing, regular axis `target`.

    Each target point averages the source points under a tent as wide as the coarser of the
    two steps: plain linear interpolation when the target is finer, an anti-aliased average
    when it is coarser. Targets beyond the source take its nearest end."""
    source = check_axis(source, "source axis")
    target = check_axis(target, "target axis")
    source_step = float(np.median(np.diff(source)))
    target_step = float(target[1] - target[0])
    clipped = np.clip(target, source[0], source[-1])
    # Near the axis's ends the tent shrinks to stay symmetric (down to plain linear
    # interpolation at the ends): a one-sided average would bias the edge cells.
    room = np.minimum(clipped - source[0], source[-1] - clipped)
    width = np.maximum(source_step, np.minimum(max(source_step, target_step), room))
    weights = np.maximum(0.0, 1.0 - np.abs(source[None, :] - clipped[:, None]) / width[:, None])
    totals = weights.sum(axis=1, keepdims=True)
    # A target between two sources farther apart than `width` (an irregular source axis):
    # interpolate between its neighbours instead.
    empty = totals[:, 0] <= 0
    if np.any(empty):
        for row in np.flatnonzero(empty):
            right = int(np.searchsorted(source, clipped[row]))
            left = right - 1
            span = source[right] - source[left]
            weights[row, left] = (source[right] - clipped[row]) / span
            weights[row, right] = (clipped[row] - source[left]) / span
        totals = weights.sum(axis=1, keepdims=True)
    return weights / totals


def to_canonical(
    image: np.ndarray,
    frequencies: np.ndarray,
    velocities: np.ndarray,
    grid: CanonicalGrid,
    f_range: tuple[float, float] | None = None,
    v_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, CanonicalAxes]:
    """The (n_f, n_v) image on the canonical grid, spanning its own axes by default or the
    given sub-ranges (a zoom), and the canonical axes."""
    frequencies = check_axis(frequencies, "frequencies")
    velocities = check_axis(velocities, "velocities")
    image = np.asarray(image, dtype=np.float32)
    if image.shape != (frequencies.size, velocities.size):
        raise ValueError(
            f"image shape {image.shape} does not match the axes "
            f"({frequencies.size}, {velocities.size})"
        )
    f_range = f_range or (float(frequencies[0]), float(frequencies[-1]))
    v_range = v_range or (float(velocities[0]), float(velocities[-1]))
    axes = grid.axes(f_range, v_range)
    rows = resampling_matrix(frequencies, axes.frequencies).astype(np.float32)
    columns = resampling_matrix(velocities, axes.velocities).astype(np.float32)
    return rows @ image @ columns.T, axes


def from_canonical(values: np.ndarray, axes_values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Values along a canonical axis interpolated at `targets` (NaN kept: a target next to a
    NaN value is NaN)."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    out = np.interp(targets, axes_values, np.where(finite, values, 0.0))
    near_nan = np.interp(targets, axes_values, (~finite).astype(np.float64)) > 0
    out[near_nan] = np.nan
    return out
