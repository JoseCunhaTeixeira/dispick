"""Classical pickers to compare against, scored like the network.

`maximum` is sigpipe's `pick_curves` without bounds: each frequency's brightest velocity (the
last one on a tie, as sigpipe takes it), outliers from a 5-point median replaced, a
Savitzky-Golay smoothing; searched above the aliasing limit (2 x spacing), as PACo's picker
starts. It picks every frequency: it has no notion of where the ridge fades.
"""

import numpy as np
from scipy.signal import medfilt, savgol_filter

from dispick.evaluation.benchmark import Pick
from dispick.synthesis.sample import SyntheticSample


def maximum(sample: SyntheticSample) -> Pick:
    fs, vs, image = sample.frequencies, sample.velocities, sample.fv_map.astype(np.float64)
    lowest = 2.0 * sample.geometry.spacing * fs
    rows = np.flatnonzero((fs > 0) & (lowest < vs[-1]))
    velocities = np.full(fs.size, np.nan)
    picked = np.zeros(fs.size, dtype=bool)
    if rows.size < 2:
        return Pick(velocities=velocities, picked=picked)
    values = np.empty(rows.size)
    for k, row in enumerate(rows):
        column = np.where(vs >= lowest[row], image[row], -np.inf)
        values[k] = vs[np.flatnonzero(column == np.max(column))[-1]]
    if values.size >= 5:
        median = medfilt(values, kernel_size=5)
        residual = np.abs(values - median)
        outliers = residual > 2.5 * np.median(residual)
        if np.any(outliers) and not np.all(outliers):
            values[outliers] = np.interp(fs[rows][outliers], fs[rows][~outliers], values[~outliers])
        window = values.size // 2 + 1 if (values.size // 2) % 2 == 0 else values.size // 2
        if window > 2:
            values = savgol_filter(values, window_length=window, polyorder=2)
    velocities[rows] = values
    picked[rows] = True
    return Pick(velocities=velocities, picked=picked)


def maximum_method(samples: list[SyntheticSample]) -> list[Pick]:
    return [maximum(sample) for sample in samples]
