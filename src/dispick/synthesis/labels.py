"""Where M0 can be picked in a synthetic image, and whether the image is worth picking.

A synthetic image's truth is its modes' exact velocities, but a pick is only as good as what the
image shows. M0 counts as pickable at a frequency when the image has a ridge peak within
`tolerance` of M0's velocity, standing clearly above the noise floor, not dwarfed by the
column's highest peak, at a wavelength the array does not alias. The network learns to say
where that holds: its presence output is a probability that a pick there is right.

The image's own labels, for the quality head:

- `pickable`: some continuous stretch of pickable M0 spans at least `min_octaves` of wavelength;
- `quality`: the widest such span over `quality_octaves` (capped at 1), since what a curve is
  worth to an inversion is the depth range its wavelengths cover;
- `higher_mode_share`: the share of M0's columns where the column's highest peak is a higher
  mode, a warning that the image invites mode jumps.
"""

from itertools import pairwise

import numpy as np

from dispick.features import Geometry
from dispick.synthesis.config import LabelConfig
from dispick.synthesis.sample import ImageLabels


def local_peaks(image: np.ndarray) -> np.ndarray:
    """(n_f, n_v) mask of each column's interior local maxima along velocity (the first cell of
    a plateau counts)."""
    peaks = np.zeros(image.shape, dtype=bool)
    if image.shape[1] >= 3:
        peaks[:, 1:-1] = (image[:, 1:-1] > image[:, :-2]) & (image[:, 1:-1] >= image[:, 2:])
    return peaks


def _top_peak(image_row: np.ndarray, candidates: np.ndarray) -> int | None:
    if candidates.size == 0:
        return None
    return int(candidates[np.argmax(image_row[candidates])])


def m0_visibility(
    image: np.ndarray,
    frequencies: np.ndarray,
    velocities: np.ndarray,
    c0: np.ndarray,
    geometry: Geometry,
    config: LabelConfig,
) -> np.ndarray:
    """(n_f,) mask of the columns where M0 (velocities `c0`, NaN where absent) is pickable."""
    floor = geometry.noise_floor
    step = float(np.median(np.diff(velocities)))
    peaks = local_peaks(image)
    lowest = config.alias_wavelength * geometry.spacing * frequencies
    visible = np.zeros(frequencies.size, dtype=bool)
    for i, (f, c) in enumerate(zip(frequencies, c0, strict=True)):
        if not np.isfinite(c) or f <= 0 or not velocities[0] <= c <= velocities[-1]:
            continue
        if c < lowest[i]:
            continue
        candidates = np.flatnonzero(peaks[i] & (velocities >= lowest[i]))
        if candidates.size == 0:
            continue
        nearest = int(candidates[np.argmin(np.abs(velocities[candidates] - c))])
        tolerance = max(config.tolerance, 1.5 * step / c)
        if abs(velocities[nearest] - c) > tolerance * c:
            continue
        height = float(image[i, nearest])
        if height < floor + config.margin * (1.0 - floor):
            continue
        top = _top_peak(image[i], candidates)
        if top is not None and height < config.relative_height * float(image[i, top]):
            continue
        visible[i] = True
    return clean_runs(visible, config)


def min_run(n_columns: int, config: LabelConfig) -> int:
    """The shortest stretch that counts, in columns: `min_run`, less on very coarse images."""
    return max(1, min(config.min_run, n_columns // 4))


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) index pairs, both included, of the mask's runs of True."""
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def clean_runs(mask: np.ndarray, config: LabelConfig) -> np.ndarray:
    """The mask with short gaps closed (a pick bridges a column that noise spoiled) and short
    stretches dropped (a lucky peak is not a ridge)."""
    out = mask.astype(bool).copy()
    gap = max(1, round(config.gap * mask.size))
    stretches = runs(out)
    for (_, end), (start, _) in pairwise(stretches):
        if start - end - 1 <= gap:
            out[end + 1 : start] = True
    shortest = min_run(mask.size, config)
    for start, end in runs(out):
        if end - start + 1 < shortest:
            out[start : end + 1] = False
    return out


def image_labels(
    image: np.ndarray,
    frequencies: np.ndarray,
    velocities: np.ndarray,
    curves: np.ndarray,
    visible: np.ndarray,
    geometry: Geometry,
    config: LabelConfig,
) -> ImageLabels:
    """The image-level labels, from the pickable mask and the true curves (n_modes, n_f)."""
    c0 = curves[0]
    span = 0.0
    shortest = min_run(visible.size, config)
    for start, end in runs(visible):
        if end - start + 1 < shortest:
            continue
        long = c0[start] / frequencies[start]
        short = c0[end] / frequencies[end]
        span = max(span, float(np.log2(max(long, short) / min(long, short))))
    pickable = span >= config.min_octaves
    return ImageLabels(
        pickable=bool(pickable),
        quality=float(min(1.0, span / config.quality_octaves)) if pickable else 0.0,
        higher_mode_share=_higher_mode_share(
            image, frequencies, velocities, curves, geometry, config
        ),
    )


def _higher_mode_share(
    image: np.ndarray,
    frequencies: np.ndarray,
    velocities: np.ndarray,
    curves: np.ndarray,
    geometry: Geometry,
    config: LabelConfig,
) -> float:
    peaks = local_peaks(image)
    lowest = config.alias_wavelength * geometry.spacing * frequencies
    counted = higher = 0
    for i, f in enumerate(frequencies):
        c0 = curves[0, i]
        if not np.isfinite(c0) or f <= 0 or not velocities[0] <= c0 <= velocities[-1]:
            continue
        if c0 < lowest[i]:
            continue
        candidates = np.flatnonzero(peaks[i] & (velocities >= lowest[i]))
        top = _top_peak(image[i], candidates)
        if top is None:
            continue
        counted += 1
        v = velocities[top]
        if abs(v - c0) <= config.tolerance * c0:
            continue
        others = curves[1:, i]
        others = others[np.isfinite(others)]
        if others.size and np.min(np.abs(others - v) / others) <= config.tolerance:
            higher += 1
    return higher / counted if counted else 0.0
