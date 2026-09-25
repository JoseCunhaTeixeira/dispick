"""The phase-shift transform (Park et al., 1998) as sigpipe computes it, from spectra.

sigpipe's `phase_shift` starts from records in time; the synthesis builds spectra directly, at
the image's frequencies. Step for step what sigpipe does: each trace weighted by sqrt(offset)
(so a zero-offset trace vanishes), each spectrum normalized to unit amplitude, steered by
exp(+i 2 pi f x / v), summed over traces, and divided by the number of distinct offsets.
"""

import numpy as np


def phase_shift(
    spectra: np.ndarray,
    frequencies: np.ndarray,
    offsets: np.ndarray,
    velocities: np.ndarray,
) -> np.ndarray:
    """The (n_f, n_v) phase-shift image of the (N, n_f) spectra recorded at `offsets` (m).

    Regularly spaced offsets take a fast path (Horner's scheme on exp(i 2 pi f dx / v)),
    others the direct sum."""
    spectra = np.asarray(spectra)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    if spectra.ndim != 2 or spectra.shape != (offsets.size, frequencies.size):
        raise ValueError(
            f"spectra must be (n_offsets, n_frequencies) = {(offsets.size, frequencies.size)}, "
            f"got {spectra.shape}"
        )
    weighted = spectra * np.sqrt(offsets)[:, None]
    unit = (weighted / (np.abs(weighted) + 1e-12)).astype(np.complex128)
    order = np.argsort(offsets, kind="stable")
    sorted_offsets = offsets[order]
    steps = np.diff(sorted_offsets)
    if steps.size and steps[0] > 0 and np.allclose(steps, steps[0], rtol=1e-6, atol=0.0):
        total = _horner(unit[order], frequencies, float(steps[0]), velocities)
    else:
        total = _direct(unit, frequencies, offsets, velocities)
    return (total / np.unique(offsets).size).astype(np.float32)


def _horner(
    unit: np.ndarray, frequencies: np.ndarray, spacing: float, velocities: np.ndarray
) -> np.ndarray:
    """|sum_n u_n exp(i 2 pi f (x0 + n dx) / v)| = |sum_n u_n z^n| with z = exp(i 2 pi f dx / v):
    the first offset's phase is common to all terms and drops out of the magnitude."""
    z = np.exp(1j * (2 * np.pi * spacing) * frequencies[:, None] / (velocities[None, :] + 1e-12))
    total = np.broadcast_to(unit[-1][:, None], z.shape).astype(np.complex128)
    for n in range(unit.shape[0] - 2, -1, -1):
        total = total * z + unit[n][:, None]
    return np.abs(total)


def _direct(
    unit: np.ndarray, frequencies: np.ndarray, offsets: np.ndarray, velocities: np.ndarray
) -> np.ndarray:
    total = np.empty((frequencies.size, velocities.size))
    phase = 2 * np.pi * offsets[:, None] * frequencies[None, :]
    for j, velocity in enumerate(velocities):
        total[:, j] = np.abs(np.sum(unit * np.exp(1j * phase / (velocity + 1e-12)), axis=0))
    return total
