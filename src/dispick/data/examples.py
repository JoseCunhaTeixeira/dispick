"""Synthetic samples on the canonical grid, with the targets the network learns."""

from dataclasses import dataclass

import numpy as np

from dispick.features import Geometry, input_channels
from dispick.grid import CanonicalAxes, CanonicalGrid, frequency_weights, to_canonical
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
    """The sample resampled onto `grid`, with the targets of its first `n_modes` modes.

    The targets are made of the image's own columns, weighted as the canonical rows are: a
    row copying a column (a coarse image) gets that column's velocity and label, a row
    averaging several (a fine image) their average velocity and their majority label. So a
    target always sits on the ridge the row shows."""
    image, axes = to_canonical(sample.fv_map, sample.frequencies, sample.velocities, grid)
    weights = frequency_weights(sample.frequencies, axes)  # (F, n_f)
    curves = np.full((n_modes, sample.frequencies.size), np.nan)
    count = min(n_modes, sample.curves.shape[0])
    curves[:count] = sample.curves[:count]
    missing = weights @ (~np.isfinite(curves)).T.astype(np.float64) > 0  # (F, n_modes)
    rows = weights @ np.nan_to_num(curves, nan=0.0).T
    rows[missing] = np.nan
    bins = axes.bin_of_velocity(rows.T)
    bins[~((bins >= 0) & (bins <= grid.n_velocities - 1))] = np.nan
    presence = (weights @ sample.visible.astype(np.float64) >= 0.5) & np.isfinite(bins[0])
    presence &= axes.frequencies > 0
    return TrainingExample(
        image=image,
        f_range=axes.f_range,
        v_range=axes.v_range,
        geometry=sample.geometry,
        target_bins=bins.astype(np.float32),
        presence=presence.astype(np.float32),
        image_targets=sample.labels.as_array(),
    )
