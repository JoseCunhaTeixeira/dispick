"""Synthetic samples on the canonical grid, with the targets the network learns."""

from dataclasses import dataclass

import numpy as np

from dispick.features import Geometry, input_channels
from dispick.grid import CanonicalAxes, CanonicalGrid, to_canonical
from dispick.synthesis.sample import SyntheticSample


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One image on the canonical grid and its targets.

    `image` holds the raw phase-shift values, not the network's input (see `inputs`): it
    stores compactly and lets the geometry be dropped at training time. `target_bins` are the
    modes' fractional velocity bins at each canonical frequency (NaN where a mode is absent
    or off the image), `presence` where M0 is pickable, `image_targets` the image's labels."""

    image: np.ndarray  # (F, V) float32
    f_range: tuple[float, float]
    v_range: tuple[float, float]
    geometry: Geometry
    target_bins: np.ndarray  # (n_modes, F) float32
    presence: np.ndarray  # (F,) float32, 0 or 1
    image_targets: np.ndarray  # (3,) float32

    def axes(self) -> CanonicalAxes:
        grid = CanonicalGrid(self.image.shape[0], self.image.shape[1])
        return grid.axes(self.f_range, self.v_range)

    def inputs(self, with_geometry: bool = True) -> np.ndarray:
        """The network's (C, F, V) input; without the geometry, as for an unknown array."""
        return input_channels(self.image, self.axes(), self.geometry if with_geometry else None)


def make_example(sample: SyntheticSample, grid: CanonicalGrid, n_modes: int) -> TrainingExample:
    """The sample resampled onto `grid`, with the targets of its first `n_modes` modes."""
    image, axes = to_canonical(sample.fv_map, sample.frequencies, sample.velocities, grid)
    curves = np.full((n_modes, grid.n_frequencies), np.nan)
    known = sample.curves_at(axes.frequencies)
    count = min(n_modes, known.shape[0])
    curves[:count] = known[:count]
    bins = axes.bin_of_velocity(curves)
    bins[~((bins >= 0) & (bins <= grid.n_velocities - 1))] = np.nan
    # Each canonical frequency takes the label of the nearest column of the image.
    nearest = np.clip(
        np.searchsorted(sample.frequencies, axes.frequencies), 1, sample.frequencies.size - 1
    )
    left = sample.frequencies[nearest - 1]
    right = sample.frequencies[nearest]
    nearest = np.where(axes.frequencies - left <= right - axes.frequencies, nearest - 1, nearest)
    presence = sample.visible[nearest] & np.isfinite(bins[0]) & (axes.frequencies > 0)
    return TrainingExample(
        image=image,
        f_range=axes.f_range,
        v_range=axes.v_range,
        geometry=sample.geometry,
        target_bins=bins.astype(np.float32),
        presence=presence.astype(np.float32),
        image_targets=sample.labels.as_array(),
    )
