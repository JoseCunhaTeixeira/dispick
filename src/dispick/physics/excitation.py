"""How strongly a vertical force at the surface excites each Rayleigh mode on vertical geophones.

The medium response of Aki & Richards (2002, eq. 7.150): mode m weighs

    A_m(f) = uz_m(0)² / (c_m U_m I_m),    I_m = ∫ rho (ur_m² + uz_m²) dz,

at every offset alike (the offset enters through the Hankel function, in the wavefield). The
ratio does not depend on how the eigenfunctions are normalized. Which mode dominates decides
which ridge is brightest in the image: M0 on steadily dispersive ground, higher modes above a
stiff crust or a strong bedrock contrast.
"""

import numpy as np
from disba import DispersionError, EigenFunction

from dispick.physics.dispersion import NormalizedModel


def group_velocity(c: np.ndarray, f_hat: np.ndarray) -> np.ndarray:
    """U = c / (1 - d ln c / d ln f) along the last axis of `c` (NaN kept), on the increasing
    grid `f_hat`."""
    log_f = np.log(f_hat)
    out = np.full_like(c, np.nan, dtype=np.float64)
    for index in np.ndindex(c.shape[:-1]):
        row = c[index]
        finite = np.isfinite(row)
        if finite.sum() < 2:
            continue
        slope = np.gradient(np.log(row[finite]), log_f[finite])
        out[(*index, finite)] = row[finite] / np.clip(1.0 - slope, 0.05, None)
    return out


def sublayered(
    model: NormalizedModel, top: float, bottom: float, growth: float = 1.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The model cut into layers growing geometrically from `top` thick at the surface down to
    `bottom` (the half-space cut too), the model's own interfaces kept: disba samples the
    eigenfunctions at layer tops only."""
    tops = np.concatenate([[0.0], np.cumsum(model.thickness[:-1])])
    edges = [0.0]
    step = top
    while edges[-1] < bottom:
        edges.append(edges[-1] + step)
        step *= growth
    edges = np.union1d(np.asarray(edges), tops)
    edges = edges[np.concatenate([[True], np.diff(edges) > 1e-9 * max(bottom, 1.0)])]
    thickness = np.diff(edges)
    layer = np.searchsorted(tops, edges[:-1] + thickness / 2, side="right") - 1
    return (
        np.append(thickness, 1.0),
        np.append(model.vs[layer], model.vs[-1]),
        np.append(model.vp[layer], model.vp[-1]),
        np.append(model.rho[layer], model.rho[-1]),
    )


def vertical_excitation(
    model: NormalizedModel,
    f_hat: np.ndarray,
    c: np.ndarray,
    n_samples: int = 24,
    dc: float = 0.001,
) -> np.ndarray:
    """Relative excitation (n_modes, n_f) of the modes `c` (normalized phase velocities on the
    increasing grid `f_hat`): each column divided by its largest mode, NaN where a mode does not
    exist. Computed at `n_samples` log-spaced frequencies and interpolated in log-log: the
    response varies smoothly, away from cut-offs."""
    n_modes, n_f = c.shape
    group = group_velocity(c, f_hat)
    samples = np.unique(np.geomspace(f_hat[0], f_hat[-1], n_samples))
    c_samples = np.array([np.interp(samples, f_hat, row, left=np.nan) for row in c])
    u_samples = np.array([np.interp(samples, f_hat, row, left=np.nan) for row in group])
    # Where a mode starts between two samples, np.interp would blend it with NaN: mark absent.
    for mode in range(n_modes):
        first = np.flatnonzero(np.isfinite(c[mode]))
        if first.size:
            c_samples[mode, samples < f_hat[first[0]]] = np.nan

    wavelengths = c_samples / samples
    shortest = float(np.nanmin(wavelengths))
    longest = float(np.nanmax(wavelengths))
    thickness, vs_layers, vp_layers, rho_layers = sublayered(
        model, top=shortest / 12, bottom=1.0 + 2.5 * longest
    )
    engine = EigenFunction(thickness, vp_layers, vs_layers, rho_layers, dc=dc)

    response = np.full((n_modes, samples.size), np.nan)
    for j, f in enumerate(samples):
        for mode in range(n_modes):
            phase, group_j = c_samples[mode, j], u_samples[mode, j]
            if not (np.isfinite(phase) and np.isfinite(group_j)):
                continue
            try:
                eigen = engine(1.0 / f, mode=mode, wave="rayleigh")
            except DispersionError:
                continue
            energy = _energy(eigen.ur, eigen.uz, thickness, rho_layers, vs_layers, phase, f)
            if np.isfinite(energy) and energy > 0 and eigen.uz.size:
                response[mode, j] = float(eigen.uz[0]) ** 2 / (phase * group_j * energy)

    out = np.full((n_modes, n_f), np.nan)
    log_f, log_samples = np.log(f_hat), np.log(samples)
    for mode in range(n_modes):
        known = np.isfinite(response[mode]) & (response[mode] > 0)
        exists = np.isfinite(c[mode])
        if known.sum() >= 2:
            out[mode, exists] = np.exp(
                np.interp(log_f[exists], log_samples[known], np.log(response[mode, known]))
            )
        elif known.sum() == 1:
            out[mode, exists] = response[mode, known][0]
    peak = np.nanmax(np.where(np.isfinite(out), out, -np.inf), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(out) & (peak > 0), out / peak, np.nan)


def _energy(
    ur: np.ndarray,
    uz: np.ndarray,
    thickness: np.ndarray,
    rho: np.ndarray,
    vs: np.ndarray,
    phase: float,
    f: float,
) -> float:
    """∫ rho (ur² + uz²) dz: per layer by the trapezoid rule (displacements are continuous
    across interfaces, density is not), plus the half-space's tail beyond the last sample,
    decaying at least as fast as its shear part, exp(-k z sqrt(1 - c²/vs²))."""
    power = ur**2 + uz**2
    n = power.size
    inner = float(np.sum(rho[: n - 1] * thickness[: n - 1] * (power[:-1] + power[1:]) / 2))
    ratio = phase / vs[-1]
    if ratio >= 1.0:
        return float("inf")
    decay = 2 * np.pi * f / phase * np.sqrt(1.0 - ratio**2)
    return inner + float(rho[-1] * power[-1] / (2 * decay))
