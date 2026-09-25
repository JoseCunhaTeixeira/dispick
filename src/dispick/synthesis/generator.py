"""Synthetic dispersion images, each with its truth.

One image: a model from the bank, scaled to a random receiver array so that its dispersion
falls where the array resolves it (or, on purpose, partly beyond); axes drawn as users set
them (fmax from the shortest wavelength wanted, often round numbers, fmin often 0, sigpipe's
frequency steps and 1000 velocities, PAC's vmin of 1 m/s); records synthesized with the modes
and the defects of real data; the image computed by sigpipe's phase shift; and the labels.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dispick.features import Geometry
from dispick.physics.bank import BankEntry, ModalBank
from dispick.physics.earth import log_uniform
from dispick.synthesis.acquisition import Array, sample_array
from dispick.synthesis.config import NoisePrior, SynthesisConfig
from dispick.synthesis.labels import image_labels, m0_visibility
from dispick.synthesis.sample import ImageLabels, SyntheticSample
from dispick.synthesis.transform import phase_shift
from dispick.synthesis.wavefield import (
    Heterogeneity,
    Modes,
    coherent_noise,
    dc_values,
    random_noise,
    signal_rms,
    snr_profile,
    surface_waves,
)

SCENARIOS: tuple[str, ...] = ("survey", "noise_only", "coherent_only", "off_range")

type CurveFunction = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]

_NICE = np.array([1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])


def nice(value: float, direction: str = "nearest") -> float:
    """`value` rounded to 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6 or 8 times a power of ten."""
    if value <= 0:
        return value
    power = 10.0 ** np.floor(np.log10(value))
    candidates = _NICE * power
    if direction == "up":
        return float(candidates[np.searchsorted(candidates, value * (1 - 1e-9))])
    if direction == "down":
        return float(candidates[np.searchsorted(candidates, value * (1 + 1e-9), side="right") - 1])
    return float(candidates[np.argmin(np.abs(np.log(candidates / value)))])


@dataclass(frozen=True, slots=True)
class TraceDefects:
    """What is wrong with each receiver, the same for every record of a stack."""

    coupling: np.ndarray
    dead: np.ndarray
    flipped: np.ndarray
    delays: np.ndarray

    @classmethod
    def draw(
        cls,
        n: int,
        top_frequency: float,
        prior: NoisePrior,
        rng: np.random.Generator,
        events: list[str],
    ) -> TraceDefects:
        coupling = rng.lognormal(0.0, rng.uniform(*prior.coupling), n)
        dead = np.zeros(n, dtype=bool)
        if rng.random() < prior.dead_probability:
            dead = rng.random(n) < rng.uniform(*prior.dead_fraction)
            if dead.any():
                events.append("dead")
        flipped = np.zeros(n, dtype=bool)
        if rng.random() < prior.polarity_probability:
            flipped = rng.random(n) < rng.uniform(*prior.polarity_fraction)
            if flipped.any():
                events.append("polarity")
        delays = np.zeros(n)
        if rng.random() < prior.statics_probability and top_frequency > 0:
            delays = rng.normal(0.0, rng.uniform(*prior.statics) / top_frequency, n)
            events.append("statics")
        return cls(coupling=coupling, dead=dead, flipped=flipped, delays=delays)

    def apply(
        self,
        spectra: np.ndarray,
        frequencies: np.ndarray,
        rms: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        out = spectra * self.coupling[:, None]
        if self.dead.any():
            out[self.dead] = random_noise(rms, np.zeros_like(rms), int(self.dead.sum()), rng)
        out[self.flipped] *= -1.0
        return out * np.exp(-2j * np.pi * frequencies[None, :] * self.delays[:, None])


class SyntheticGenerator:
    """Draws `SyntheticSample`s from a modal bank. Deterministic given the random generator."""

    def __init__(self, bank: ModalBank, config: SynthesisConfig | None = None) -> None:
        self.bank = bank
        self.config = config or SynthesisConfig()

    def sample(self, rng: np.random.Generator) -> SyntheticSample:
        config = self.config
        scenario = self._scenario(rng)
        entry = self.bank[int(rng.integers(len(self.bank)))]
        array = sample_array(config.array, rng)
        geometry = array.geometry
        velocity_scale = log_uniform(rng, config.scale.velocity_scale)
        central = 2.0 * np.sqrt(geometry.spacing * geometry.aperture)
        length_scale = central * log_uniform(rng, config.scale.depth_over_wavelength) / entry.length
        heterogeneity = None
        if rng.random() < config.wavefield.heterogeneity_probability:
            heterogeneity = Heterogeneity(
                split=float(
                    array.offsets[0]
                    + rng.uniform(0.2, 0.8) * (array.offsets[-1] - array.offsets[0])
                ),
                contrast=log_uniform(rng, config.wavefield.heterogeneity_contrast),
            )
        # The records follow the ground under the array's first part (and the change past the
        # split); the truth is what the phase shift reads, the ground's average along the array.
        factor = heterogeneity.factor(array.offsets) if heterogeneity is not None else 1.0

        def curves(frequencies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            c, a = entry.curves(frequencies, velocity_scale, length_scale)
            return c * factor, a

        frequencies = self._frequencies(entry, curves, velocity_scale, length_scale, geometry, rng)
        positive = frequencies > 0
        c_image = np.full((entry.n_modes, frequencies.size), np.nan)
        a_image = np.zeros((entry.n_modes, frequencies.size))
        c_image[:, positive], a_image[:, positive] = curves(frequencies[positive])
        c_first = c_image / factor
        velocities = self._velocities(c_image[0], frequencies, geometry, scenario, rng)

        events: list[str] = []
        info: dict[str, str | float | int] = {
            "scenario": scenario,
            "kind": array.kind,
            "family": str(entry.model.family),
            "bank_index": entry.index,
            "n_receivers": geometry.n_receivers,
            "spacing": geometry.spacing,
            "heterogeneity": int(heterogeneity is not None),
        }
        # Body waves cross the array at the layers' P and (refracted) S velocities.
        body = np.concatenate([entry.model.vp, entry.model.vs[1:]]) * velocity_scale
        image = self._image(
            array, frequencies, velocities, c_first, a_image, body, scenario, heterogeneity,
            rng, events, info,
        )  # fmt: skip
        info["events"] = ",".join(sorted(set(events)))
        info["n_frequencies"] = int(frequencies.size)
        info["n_velocities"] = int(velocities.size)

        dense = self._dense_frequencies(frequencies)
        dense_curves, _ = curves(dense)
        if scenario in ("noise_only", "coherent_only"):
            # No surface waves in the records: no curve is there to find, nor to learn.
            c_image = np.full_like(c_image, np.nan)
            dense_curves = np.full_like(dense_curves, np.nan)
            visible = np.zeros(frequencies.size, dtype=bool)
            labels = ImageLabels(pickable=False, quality=0.0, higher_mode_share=0.0)
        else:
            # The phase shift weighs out a zero-offset trace (a passive virtual source's).
            traces = int(np.count_nonzero(array.offsets > 0))
            visible = m0_visibility(
                image, frequencies, velocities, c_image[0], geometry, config.labels, traces
            )
            labels = image_labels(
                image, frequencies, velocities, c_image, visible, geometry, config.labels
            )
        return SyntheticSample(
            fv_map=image,
            frequencies=frequencies,
            velocities=velocities,
            offsets=array.offsets,
            geometry=geometry,
            curves=c_image,
            dense_frequencies=dense,
            dense_curves=dense_curves,
            visible=visible,
            labels=labels,
            info=info,
        )

    def _scenario(self, rng: np.random.Generator) -> str:
        prior = self.config.scenarios
        draw = rng.random()
        for name, probability in (
            ("noise_only", prior.noise_only_probability),
            ("coherent_only", prior.coherent_only_probability),
            ("off_range", prior.off_range_probability),
        ):
            if draw < probability:
                return name
            draw -= probability
        return "survey"

    def _frequencies(
        self,
        entry: BankEntry,
        curves: CurveFunction,
        velocity_scale: float,
        length_scale: float,
        geometry: Geometry,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """sigpipe's frequency axis: multiples of df = sampling rate / samples between fmin and
        fmax, fmax set by the shortest M0 wavelength wanted."""
        prior = self.config.axes
        scale = velocity_scale * entry.velocity / (length_scale * entry.length)
        wide = np.geomspace(entry.f_hat[0] / 20, entry.f_hat[-1] * 20, 1024) * scale
        wavelengths = curves(wide)[0][0] / wide  # decreasing with frequency
        order = np.argsort(wavelengths)

        def frequency_at(wavelength: float) -> float:
            return float(np.interp(wavelength, wavelengths[order], wide[order]))

        fmax = frequency_at(geometry.spacing * log_uniform(rng, prior.shortest_wavelength))
        if rng.random() < prior.round_fmax_probability:
            fmax = nice(fmax)
        fmin = 0.0
        if rng.random() >= prior.fmin_zero_probability:
            fmin = frequency_at(geometry.aperture * log_uniform(rng, prior.longest_wavelength))
            fmin = min(fmin, 0.6 * fmax)
            if rng.random() < prior.round_fmax_probability:
                fmin = nice(fmin, "down")
        if rng.random() < prior.coarse_probability:
            count = int(
                rng.integers(prior.coarse_n_frequencies[0], prior.coarse_n_frequencies[1] + 1)
            )
        else:
            count = round(log_uniform(rng, prior.n_frequencies))
        step = (fmax - fmin) / max(count - 1, 1)
        first = int(np.ceil(fmin / step - 1e-9))
        last = int(np.floor(fmax / step + 1e-9))
        if last - first < 2:
            last = first + 2
        return np.arange(first, last + 1) * step

    def _velocities(
        self,
        c0: np.ndarray,
        frequencies: np.ndarray,
        geometry: Geometry,
        scenario: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """sigpipe's velocity axis, linspace(vmin, vmax, nv), vmax above M0 where the array
        resolves it (or, for `off_range`, missing M0 on purpose)."""
        prior = self.config.axes
        positive = (frequencies > 0) & np.isfinite(c0)
        wavelength = np.where(positive, c0 / np.where(positive, frequencies, 1.0), np.inf)
        resolved = positive & (wavelength <= 3.0 * geometry.aperture)
        top = float(np.max(c0[resolved])) if resolved.any() else float(np.nanmax(c0[positive]))
        bottom = float(np.nanmin(c0[positive]))
        if scenario == "off_range" and rng.random() < 0.5:
            vmin = float(rng.choice([0.0, 1.0]))
            vmax = max(bottom * rng.uniform(0.3, 0.85), vmin + 10.0)
        elif scenario == "off_range":
            vmin = float(np.nanmax(c0[positive])) * rng.uniform(1.15, 2.0)
            vmax = vmin * rng.uniform(1.5, 4.0)
        else:
            vmax = top * log_uniform(rng, prior.vmax_over_c)
            if rng.random() < prior.round_v_probability:
                vmax = nice(vmax, "up")
            if rng.random() < prior.vmin_low_probability:
                vmin = 0.0 if rng.random() < 0.15 else 1.0
            else:
                vmin = bottom * rng.uniform(*prior.vmin_over_c)
                if rng.random() < prior.round_v_probability:
                    vmin = nice(vmin, "down")
            # Some of M0 inside, and a range as wide as users pick them (vmax/vmin >= 1.6).
            vmax = max(vmax, 1.1 * bottom, 1.6 * vmin, vmin + 0.4 * top)
        if rng.random() < prior.default_nv_probability:
            count = 1000
        else:
            count = round(log_uniform(rng, prior.n_velocities))
        budget = prior.max_cells // max(1, frequencies.size * geometry.n_receivers)
        count = int(max(64, min(count, budget)))
        return np.linspace(vmin, vmax, count)

    def _modes(
        self,
        c_image: np.ndarray,
        a_image: np.ndarray,
        frequencies: np.ndarray,
        silent: bool,
        rng: np.random.Generator,
        info: dict[str, str | float | int],
    ) -> Modes:
        prior = self.config.wavefield
        n_modes = c_image.shape[0]
        physical = rng.random() < prior.physical_excitation_probability
        info["physical_excitation"] = int(physical)
        if physical:
            amplitudes = np.nan_to_num(a_image, nan=0.0)
        else:
            amplitudes = np.broadcast_to(rng.lognormal(0.0, 1.0, n_modes)[:, None], c_image.shape)
        scatter = np.exp(rng.normal(0.0, rng.uniform(*prior.mode_scatter), n_modes))
        octaves = np.log2(np.maximum(frequencies, frequencies[frequencies > 0][0]))
        drift = np.zeros_like(c_image)
        strength = rng.uniform(*prior.mode_ripple)
        for mode in range(n_modes):
            period = rng.uniform(1.0, 4.0)
            drift[mode] = strength * np.sin(
                2 * np.pi * octaves / period + rng.uniform(0, 2 * np.pi)
            )
        amplitudes = amplitudes * scatter[:, None] * np.exp(drift)
        amplitudes = np.where(np.isfinite(c_image), amplitudes, 0.0)
        if silent:
            amplitudes = np.zeros_like(amplitudes)
        q = log_uniform(rng, prior.q) * rng.lognormal(0.0, 0.2, n_modes)
        return Modes(velocities=c_image, amplitudes=amplitudes, q=q)

    def _image(
        self,
        array: Array,
        frequencies: np.ndarray,
        velocities: np.ndarray,
        c_image: np.ndarray,
        a_image: np.ndarray,
        body: np.ndarray,
        scenario: str,
        heterogeneity: Heterogeneity | None,
        rng: np.random.Generator,
        events: list[str],
        info: dict[str, str | float | int],
    ) -> np.ndarray:
        config = self.config
        silent = scenario in ("noise_only", "coherent_only")
        modes = self._modes(c_image, a_image, frequencies, silent, rng, info)
        n = array.geometry.n_receivers
        stack = 1
        if rng.random() < config.noise.stack_probability:
            stack = int(rng.integers(config.noise.stack_count[0], config.noise.stack_count[1] + 1))
        info["stack"] = stack
        snr = snr_profile(frequencies, config.noise, rng)
        info["snr_peak"] = float(np.max(snr))
        defects = TraceDefects.draw(n, float(frequencies[-1]), config.noise, rng, events)
        images = []
        signal = coherent = None
        for realization in range(stack):
            # Passive virtual shots gather other sources in each stacked window.
            if signal is None or (array.kind == "passive" and realization > 0):
                signal = surface_waves(
                    frequencies, array.true_offsets, modes, array.kind, config.wavefield, rng,
                    heterogeneity,
                )  # fmt: skip
            rms = signal_rms(signal)
            if coherent is None:
                coherent = np.zeros_like(signal)
                if scenario != "noise_only":
                    coherent = coherent_noise(
                        frequencies, array.true_offsets, rms, array.kind, body, modes,
                        config.noise, rng, events,
                    )  # fmt: skip
            noise = random_noise(rms, snr, n, rng)
            records = defects.apply(signal + coherent + noise, frequencies, rms, rng)
            if frequencies[0] == 0:
                records[:, 0] = dc_values(n, rng)
            images.append(phase_shift(records, frequencies, array.offsets, velocities))
        return np.mean(images, axis=0).astype(np.float32)

    def _dense_frequencies(self, frequencies: np.ndarray) -> np.ndarray:
        positive = frequencies[frequencies > 0]
        return np.geomspace(positive[0] / 4, frequencies[-1], self.config.dense_frequencies)
