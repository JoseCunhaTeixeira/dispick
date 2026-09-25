"""Rayleigh-wave phase velocities of every mode, with disba, checked for mode jumps.

disba searches roots in steps of `dc` km/s. Models are rescaled first (depth to the half-space
to 1 km, the half-space's Vs to 1 km/s), so the step is relative to the model's own velocities:
a near-surface model at 80 m/s would otherwise be searched in steps of 6 % of its velocity, and
close modes skipped. Dispersion is scale-invariant, so the curves are exact once scaled back.
"""

from dataclasses import dataclass

import numpy as np
from disba import DispersionError, PhaseDispersion

from dispick.physics.earth import LayeredModel


@dataclass(frozen=True, slots=True)
class NormalizedModel:
    """A model in disba's units after rescaling: lengths over `length` m, velocities over
    `velocity` m/s, densities in g/cm³."""

    thickness: np.ndarray
    vs: np.ndarray
    vp: np.ndarray
    rho: np.ndarray
    length: float  # m
    velocity: float  # m/s

    @classmethod
    def of(cls, model: LayeredModel) -> NormalizedModel:
        length = model.depth
        velocity = float(model.vs[-1])
        thickness = model.thickness / length
        thickness[-1] = 1.0  # the half-space: disba ignores it
        return cls(
            thickness=thickness,
            vs=model.vs / velocity,
            vp=model.vp / velocity,
            rho=model.rho / 1000.0,
            length=length,
            velocity=velocity,
        )

    def frequency_scale(self) -> float:
        """Hz per normalized frequency unit (f = f_hat x velocity / length)."""
        return self.velocity / self.length


def phase_velocities(
    model: NormalizedModel,
    f_hat: np.ndarray,
    n_modes: int,
    dc: float = 0.001,
) -> np.ndarray:
    """Normalized phase velocities (n_modes, n_f) at the normalized frequencies `f_hat` (any
    order, all > 0), NaN where a mode does not exist (below its cut-off). A mode disba cannot
    compute at all is NaN, and so are all the modes above it."""
    f_hat = np.asarray(f_hat, dtype=np.float64)
    if np.any(f_hat <= 0):
        raise ValueError("normalized frequencies must be > 0")
    order = np.argsort(1.0 / f_hat)  # disba wants periods in increasing order
    periods = 1.0 / f_hat[order]
    engine = PhaseDispersion(model.thickness, model.vp, model.vs, model.rho, dc=dc)
    out = np.full((n_modes, f_hat.size), np.nan)
    for mode in range(n_modes):
        try:
            result = engine(periods, mode=mode, wave="rayleigh")
        except DispersionError:
            break
        if result.period.size == 0:
            break
        # disba drops the periods where the mode does not exist: match the ones it kept.
        kept = np.searchsorted(periods, result.period)
        kept = np.clip(kept, 0, periods.size - 1)
        matched = np.isclose(periods[kept], result.period, rtol=1e-9, atol=0.0)
        values = np.full(periods.size, np.nan)
        values[kept[matched]] = result.velocity[matched]
        out[mode, order] = values
    return out


def check_modes(
    c: np.ndarray, f_hat: np.ndarray, min_step: float = 0.04, isolation: float = 3.0
) -> tuple[int, str | None]:
    """How many of the curves (n_modes, n_f) on the increasing grid `f_hat` are reliable, from
    M0 up, and why the next one is not (None when all are).

    Rayleigh modes are ordered at every frequency (M0 slowest) and continuous along frequency;
    a root search that skips a mode breaks one or the other. A mode may be steep (soft cover
    over stiff bedrock) but then its steps are steep together: a skip shows as one step of
    |d ln c| above `min_step` and `isolation` times its neighbours', or as a zigzag of two."""
    if np.any(np.diff(f_hat) <= 0):
        raise ValueError("f_hat must be increasing")
    if not np.all(np.isfinite(c[0])):
        return 0, "fundamental mode incomplete"
    if np.any(c[0] <= 0):
        return 0, "non-positive fundamental velocity"
    for mode in range(c.shape[0]):
        row = c[mode]
        finite = np.isfinite(row)
        if not finite.any():
            # No such mode in the band, nor any above it.
            return mode, None
        # A higher mode exists above its cut-off: one run, reaching the highest frequency.
        first = int(np.argmax(finite))
        if not finite[first:].all():
            return mode, f"mode {mode} has gaps"
        if (jump := _jump(np.diff(np.log(row[first:])), min_step, isolation)) is not None:
            return mode, f"mode {mode} jumps by {jump:.3f} in log velocity"
        if mode > 0:
            below = c[mode - 1, first:]
            if not np.all(np.isfinite(below)):
                return mode, f"mode {mode} exists where mode {mode - 1} does not"
            if np.any(row[first:] <= below * (1 + 1e-4)):
                return mode, f"mode {mode} is not above mode {mode - 1}"
    return c.shape[0], None


def _jump(steps: np.ndarray, min_step: float, isolation: float) -> float | None:
    """The largest step that looks like a skipped root, or None."""
    size = np.abs(steps)
    for i in np.flatnonzero(size > min_step):
        neighbours = size[max(0, i - 2) : i + 3]
        neighbours = np.delete(neighbours, min(i, 2))
        if neighbours.size and size[i] > isolation * float(np.max(neighbours)):
            return float(size[i])
        # A zigzag: two large steps of opposite signs.
        if i + 1 < steps.size and size[i + 1] > min_step and steps[i] * steps[i + 1] < 0:
            return float(size[i])
    return None


def model_curves(
    model: LayeredModel,
    f_hat: np.ndarray,
    n_modes: int,
    steps: tuple[float, ...] = (0.003, 0.0008),
) -> tuple[NormalizedModel, np.ndarray, float] | None:
    """The normalized model, its curves on `f_hat` and the root-search step that gave them;
    None when M0 itself is unreliable. M0 is the target; the higher modes only have to look
    right in the images, so the ones `check_modes` rejects are dropped (NaN) rather than the
    model, which would bias the bank against the high-contrast ground they fail on.

    The steps (relative to the slowest layer's Vs, where modes crowd) are tried in turn.
    disba traces each mode from the shortest period up, and at the shortest wavelengths the
    higher modes crowd near the top layer's Vs, where the first root count can skip one: when a
    higher mode fails, they are searched again below that band (M0 wavelengths above the top
    layer's thickness), and left NaN above it (`BankEntry.curves` holds them flat there)."""
    normalized = NormalizedModel.of(model)
    slowest = float(np.min(normalized.vs))
    best: tuple[int, np.ndarray, float] | None = None
    for step in steps:
        dc = step * slowest
        c = phase_velocities(normalized, f_hat, n_modes, dc=dc)
        reliable, _ = check_modes(c, f_hat)
        if best is None or reliable > best[0]:
            best = (reliable, c, dc)
        if reliable == n_modes:
            break
    if best is None or best[0] == 0:
        return None
    reliable, c, dc = best
    if reliable < n_modes:
        capped = _below_crowding(normalized, f_hat, c[0], n_modes, dc)
        if capped is not None and capped[0] > reliable:
            reliable, c = capped
    c = c.copy()
    c[reliable:] = np.nan
    return normalized, c, dc


def _below_crowding(
    model: NormalizedModel, f_hat: np.ndarray, c0: np.ndarray, n_modes: int, dc: float
) -> tuple[int, np.ndarray] | None:
    """The modes searched on the part of `f_hat` where M0's wavelength exceeds the top layer's
    thickness (NaN above it), and how many are reliable; None when that part is too short."""
    below = np.flatnonzero(c0 / f_hat >= model.thickness[0])
    if below.size < 32:
        return None
    stop = int(below[-1]) + 1
    c = np.full((n_modes, f_hat.size), np.nan)
    c[:, :stop] = phase_velocities(model, f_hat[:stop], n_modes, dc=dc)
    c[0] = c0
    reliable, _ = check_modes(c[:, :stop], f_hat[:stop])
    return reliable, c
