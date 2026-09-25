"""Spectra of synthetic records, receiver by receiver, at the image's frequencies.

Surface waves are a sum of Rayleigh modes. From an active shot each mode spreads as the Hankel
function H0(2)(k r) of the point-force Green's function, which keeps the near field (the ridge
reading slow at wavelengths longer than the offsets). A passive virtual shot is the correlation
of noise from many sources: plane waves from a spread of azimuths (each reads c / cos(theta),
faster than c), or a diffuse field. On top come the defects of real records: coherent events
that are not surface waves (air wave, body waves, mains hum, reflections), random noise with a
frequency-dependent signal-to-noise ratio, and trace defects (coupling, dead traces, reversed
polarity, timing errors). Only the spectra's phases and relative sizes matter: the phase shift
normalizes each trace's spectrum.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import j0, y0

from dispick.physics.earth import log_uniform
from dispick.synthesis.acquisition import Kind
from dispick.synthesis.config import NoisePrior, WavefieldPrior


@dataclass(frozen=True, slots=True)
class Modes:
    """The modes to synthesize, at the image's frequencies: phase velocities (M, n_f) in m/s
    (NaN where absent), their amplitudes (M, n_f), and each mode's quality factor."""

    velocities: np.ndarray
    amplitudes: np.ndarray
    q: np.ndarray


@dataclass(frozen=True, slots=True)
class Heterogeneity:
    """The ground changes along the array: slowness times `contrast` past offset `split`."""

    split: float
    contrast: float

    def factor(self, offsets: np.ndarray) -> float:
        """The ratio of the velocity the phase shift reads over the first part's velocity.

        The image's ridge sits at the slope of a least-squares line through the phases:
        phase = k (r + (contrast - 1) max(0, r - split)), so k_eff = k (1 + (contrast - 1) b)
        with b the slope of max(0, r - split) against r."""
        excess = np.maximum(0.0, offsets - self.split)
        spread = float(np.var(offsets))
        slope = float(np.cov(offsets, excess, bias=True)[0, 1]) / spread if spread > 0 else 0.0
        return 1.0 / (1.0 + (self.contrast - 1.0) * slope)


def hankel2(z: np.ndarray) -> np.ndarray:
    """H0(2)(z) = J0(z) - i Y0(z) for z > 0: an outgoing cylindrical wave, exp(-i z) far out."""
    return j0(z) - 1j * y0(z)


def surface_waves(
    frequencies: np.ndarray,
    offsets: np.ndarray,
    modes: Modes,
    kind: Kind,
    prior: WavefieldPrior,
    rng: np.random.Generator,
    heterogeneity: Heterogeneity | None = None,
) -> np.ndarray:
    """(N, n_f) complex spectra of the modes at `offsets` (m, the true ones)."""
    r = np.asarray(offsets, dtype=np.float64)[:, None]
    f = np.asarray(frequencies, dtype=np.float64)[None, :]
    # Slowness-weighted distance from the source: where the ground changes, the waves slow
    # down (or speed up) past the split, whichever way they cross the array.
    path = r
    if heterogeneity is not None:
        path = r + (heterogeneity.contrast - 1.0) * np.maximum(0.0, r - heterogeneity.split)
    diffuse = kind == "active" or rng.random() < prior.passive_diffuse_probability
    cosines = strengths = np.zeros(0)
    if not diffuse:
        count = int(rng.integers(prior.passive_plane_waves[0], prior.passive_plane_waves[1] + 1))
        spread = np.deg2rad(rng.uniform(*prior.passive_spread))
        azimuths = rng.normal(0.0, spread, count)
        backward = rng.random(count) < rng.uniform(*prior.passive_backward)
        azimuths[backward] += np.pi
        cosines = np.cos(azimuths)
        strengths = rng.lognormal(0.0, 0.5, count)
        strengths /= np.sqrt(np.sum(strengths**2))
    total = np.zeros((r.shape[0], f.shape[1]), dtype=np.complex128)
    for mode in range(modes.velocities.shape[0]):
        c = modes.velocities[mode][None, :]
        weight = modes.amplitudes[mode][None, :]
        valid = np.isfinite(c) & np.isfinite(weight) & (weight > 0) & (f > 0)
        if not valid.any():
            continue
        c = np.where(valid, c, 1.0)
        k = 2 * np.pi * f / c
        if diffuse:
            kr = np.maximum(k * r, 1e-9)
            wave = np.where(r > 0, hankel2(kr), 0.0)
            if heterogeneity is not None:
                wave = wave * np.exp(-1j * k * (path - r))
        else:
            wave = np.zeros_like(total)
            for cosine, strength in zip(cosines, strengths, strict=True):
                wave += strength * np.exp(-1j * k * path * cosine)
        attenuation = np.exp(-np.pi * f * r / (modes.q[mode] * c))
        total += np.where(valid, weight * attenuation * wave, 0.0)
    return total


def signal_rms(spectra: np.ndarray) -> np.ndarray:
    """Per-frequency rms over traces (1 where there is no signal at all)."""
    rms = np.sqrt(np.mean(np.abs(spectra) ** 2, axis=0))
    return np.where(rms > 0, rms, 1.0)


def smooth_bump(frequencies: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random smooth bump in log-frequency over the band, peaking at 1."""
    positive = frequencies[frequencies > 0]
    if positive.size == 0:
        return np.zeros_like(frequencies)
    log_f = np.log(np.maximum(frequencies, positive[0] / 2))
    centre = rng.uniform(np.log(positive[0]), np.log(positive[-1]))
    width = rng.uniform(0.3, 1.5)
    return np.exp(-0.5 * ((log_f - centre) / width) ** 2)


def snr_profile(frequencies: np.ndarray, prior: NoisePrior, rng: np.random.Generator) -> np.ndarray:
    """Signal-to-noise ratio (dB) along frequency: a source band, roll-offs below and above it,
    and a ripple."""
    positive = frequencies[frequencies > 0]
    snr = np.full(frequencies.size, -40.0)
    if positive.size == 0:
        return snr
    octaves = np.log2(np.maximum(frequencies, positive[0]))
    first, last = float(octaves[frequencies > 0][0]), float(octaves[-1])
    span = max(last - first, 1e-6)
    band_low = first + rng.uniform(0.0, 0.5) * span
    band_high = band_low + rng.uniform(0.3, 1.5) * (last - band_low + 1e-6)
    peak = rng.uniform(*prior.snr_peak)
    snr = (
        peak
        - rng.uniform(*prior.snr_rolloff) * np.maximum(0.0, band_low - octaves)
        - rng.uniform(*prior.snr_rolloff) * np.maximum(0.0, octaves - band_high)
    )
    for _ in range(3):
        period = rng.uniform(0.5, 4.0)
        snr += rng.uniform(*prior.snr_ripple) * np.sin(
            2 * np.pi * octaves / period + rng.uniform(0, 2 * np.pi)
        )
    return np.where(frequencies > 0, snr, -40.0)


def plane_event(
    frequencies: np.ndarray, offsets: np.ndarray, velocity: float, amplitude: np.ndarray
) -> np.ndarray:
    """A non-dispersive event crossing the array at `velocity` (inf: in phase on all traces)."""
    slowness = 0.0 if not np.isfinite(velocity) else 1.0 / velocity
    return amplitude[None, :] * np.exp(
        -2j * np.pi * frequencies[None, :] * offsets[:, None] * slowness
    )


def coherent_noise(
    frequencies: np.ndarray,
    offsets: np.ndarray,
    rms: np.ndarray,
    kind: Kind,
    body_velocities: np.ndarray,
    modes: Modes,
    prior: NoisePrior,
    rng: np.random.Generator,
    events: list[str],
) -> np.ndarray:
    """Coherent events that are not the surface waves, relative to the surface waves' rms;
    the names of those drawn are appended to `events`."""
    total = np.zeros((offsets.size, frequencies.size), dtype=np.complex128)
    if kind == "active" and rng.random() < prior.air_probability:
        amplitude = rms * log_uniform(rng, prior.air_ratio) * smooth_bump(frequencies, rng)
        total += plane_event(frequencies, offsets, rng.uniform(*prior.air_velocity), amplitude)
        events.append("air")
    if body_velocities.size and rng.random() < prior.body_probability:
        count = int(rng.integers(prior.body_count[0], prior.body_count[1] + 1))
        for velocity in rng.choice(body_velocities, size=count):
            amplitude = rms * log_uniform(rng, prior.body_ratio) * smooth_bump(frequencies, rng)
            total += plane_event(frequencies, offsets, float(velocity), amplitude)
        events.append("body")
    if rng.random() < prior.hum_probability:
        mains = float(rng.choice([50.0, 60.0]))
        width = rng.uniform(*prior.hum_width)
        ratio = log_uniform(rng, prior.hum_ratio)
        lines = np.zeros(frequencies.size)
        for harmonic in range(1, 4):
            lines += np.exp(-0.5 * ((frequencies - harmonic * mains) / width) ** 2) / harmonic
        if lines.max() > 1e-3:
            per_trace = rng.lognormal(0.0, 0.2, offsets.size)[:, None]
            jitter = np.exp(1j * rng.normal(0.0, rng.uniform(0.0, 0.5), offsets.size))[:, None]
            total += per_trace * jitter * (rms * ratio * lines)[None, :]
            events.append("hum")
    if rng.random() < prior.backward_probability:
        # A reflection off a lateral contrast beyond the array: the modes coming back.
        back = np.zeros_like(total)
        for mode in range(modes.velocities.shape[0]):
            c = modes.velocities[mode]
            valid = np.isfinite(c) & (frequencies > 0) & (modes.amplitudes[mode] > 0)
            if not valid.any():
                continue
            k = np.where(valid, 2 * np.pi * frequencies / np.where(valid, c, 1.0), 0.0)
            amplitude = np.where(valid, modes.amplitudes[mode], 0.0)
            back += amplitude[None, :] * np.exp(1j * k[None, :] * offsets[:, None])
        total += back * (log_uniform(rng, prior.backward_ratio) * rms / signal_rms(back))[None, :]
        events.append("backward")
    return total


def dc_values(n_traces: int, rng: np.random.Generator) -> np.ndarray:
    """The records' 0 Hz values. sigpipe's DC bin is real, so once each trace is normalized only
    its sign is left, and the 0 Hz column is |sum of signs| / N at every velocity: 0 for
    records demeaned exactly, up to 1 when they share an offset."""
    if rng.random() < 0.3:
        return np.zeros(n_traces)
    return rng.normal() * rng.uniform(0.0, 3.0) + rng.normal(size=n_traces)


def random_noise(
    rms: np.ndarray, snr_db: np.ndarray, n_traces: int, rng: np.random.Generator
) -> np.ndarray:
    """Complex Gaussian noise, independent per trace, at the given ratio to `rms`."""
    sigma = rms * 10 ** (-snr_db / 20) / np.sqrt(2)
    shape = (n_traces, rms.size)
    return sigma[None, :] * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def trace_defects(
    spectra: np.ndarray,
    frequencies: np.ndarray,
    rms: np.ndarray,
    prior: NoisePrior,
    rng: np.random.Generator,
    events: list[str],
) -> np.ndarray:
    """Coupling, dead traces, reversed polarities and timing errors on the (N, n_f) spectra."""
    n = spectra.shape[0]
    out = spectra * rng.lognormal(0.0, rng.uniform(*prior.coupling), n)[:, None]
    if rng.random() < prior.dead_probability:
        dead = rng.random(n) < rng.uniform(*prior.dead_fraction)
        if dead.any():
            out[dead] = random_noise(rms, np.zeros_like(rms), int(dead.sum()), rng)
            events.append("dead")
    if rng.random() < prior.polarity_probability:
        flipped = rng.random(n) < rng.uniform(*prior.polarity_fraction)
        if flipped.any():
            out[flipped] *= -1.0
            events.append("polarity")
    if rng.random() < prior.statics_probability and frequencies[-1] > 0:
        delays = rng.normal(0.0, rng.uniform(*prior.statics) / frequencies[-1], n)
        out *= np.exp(-2j * np.pi * frequencies[None, :] * delays[:, None])
        events.append("statics")
    return out
