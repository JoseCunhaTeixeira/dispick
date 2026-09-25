"""The distributions the synthetic images are drawn from.

Ranges are (low, high) pairs, drawn log-uniformly for scales and uniformly for fractions and
decibels (each field says which). The defaults aim at the MASW surveys sigpipe, PAC and PACo
process: 3 to 144 receivers, 0.1 to 5 m apart, active shots or passive virtual shots, images
from sigpipe's phase shift with its own and PAC's usual axes, and the defects real records
carry. Every dataset and model keeps the configuration it was made with.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

type Range = tuple[float, float]


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _ordered(self) -> _Section:
        for name in type(self).model_fields:
            value = getattr(self, name)
            if (
                isinstance(value, tuple)
                and len(value) == 2  # pyright: ignore[reportUnknownArgumentType]
                and all(isinstance(v, int | float) for v in value)  # pyright: ignore[reportUnknownVariableType]
                and value[0] > value[1]
            ):
                raise ValueError(f"{name}: the lower bound exceeds the upper one ({value})")
        return self


class ArrayPrior(_Section):
    # (lowest, highest, weight) bins of receiver counts, uniform within a bin.
    n_receivers: tuple[tuple[int, int, float], ...] = (
        (3, 6, 0.10),
        (7, 12, 0.22),
        (13, 24, 0.30),
        (25, 48, 0.26),
        (49, 96, 0.10),
        (97, 144, 0.02),
    )
    spacing: Range = (0.1, 5.0)  # m, log-uniform
    active_probability: float = Field(default=0.6, ge=0, le=1)
    near_offset: Range = (0.03, 3.0)  # active: nearest shot distance over the aperture, log-uniform
    irregular_probability: float = Field(default=0.03, ge=0, le=1)  # spacing varying +-30 %
    jitter_probability: float = Field(default=0.3, ge=0, le=1)
    jitter: Range = (0.0, 0.03)  # true position error std over the spacing, uniform


class ScalePrior(_Section):
    velocity_scale: Range = (0.8, 1.25)  # log-uniform, applied to the bank model
    # The model's depth to the half-space over the array's central wavelength,
    # 2 sqrt(spacing x aperture) (the geometric middle of 2 spacings and 2 apertures).
    depth_over_wavelength: Range = (0.08, 4.0)  # log-uniform


class AxesPrior(_Section):
    shortest_wavelength: Range = (
        0.5,
        12.0,
    )  # M0's wavelength at fmax over the spacing, log-uniform
    round_fmax_probability: float = Field(default=0.5, ge=0, le=1)
    fmin_zero_probability: float = Field(default=0.5, ge=0, le=1)
    longest_wavelength: Range = (0.5, 8.0)  # M0's wavelength at fmin over the aperture, log-uniform
    n_frequencies: Range = (12.0, 600.0)  # log-uniform
    coarse_probability: float = Field(default=0.08, ge=0, le=1)  # passive short segments
    coarse_n_frequencies: tuple[int, int] = (3, 12)  # uniform
    vmax_over_c: Range = (0.8, 4.0)  # over M0's top velocity in the resolved band, log-uniform
    round_v_probability: float = Field(default=0.5, ge=0, le=1)
    vmin_low_probability: float = Field(default=0.45, ge=0, le=1)  # vmin 1 m/s (or 0)
    vmin_over_c: Range = (0.1, 0.9)  # else, over M0's lowest velocity, uniform
    n_velocities: Range = (100.0, 1500.0)  # log-uniform
    default_nv_probability: float = Field(default=0.35, ge=0, le=1)  # sigpipe's 1000
    max_cells: int = Field(default=30_000_000, gt=0)  # cap on n_f x n_v x N: the compute


class WavefieldPrior(_Section):
    physical_excitation_probability: float = Field(default=0.8, ge=0, le=1)
    mode_scatter: Range = (0.2, 0.8)  # std of each mode's log-amplitude factor, uniform
    mode_ripple: Range = (0.0, 0.5)  # std of a smooth log-amplitude drift along frequency
    q: Range = (5.0, 100.0)  # quality factor, log-uniform
    passive_plane_waves: tuple[int, int] = (3, 24)  # uniform
    # std of the arrival azimuths, degrees, uniform (PAC keeps mostly endfire sources)
    passive_spread: Range = (0.0, 45.0)
    passive_backward: Range = (0.0, 0.5)  # share of plane waves from the far side, uniform
    passive_diffuse_probability: float = Field(default=0.3, ge=0, le=1)  # cylindrical, not plane
    heterogeneity_probability: float = Field(default=0.15, ge=0, le=1)
    heterogeneity_contrast: Range = (0.8, 1.25)  # slowness ratio past the split, log-uniform


class NoisePrior(_Section):
    snr_peak: Range = (-8.0, 35.0)  # dB, uniform
    snr_rolloff: Range = (3.0, 24.0)  # dB per octave outside the source's band, uniform
    snr_ripple: Range = (0.0, 4.0)  # dB, uniform
    coupling: Range = (0.0, 0.5)  # std of each trace's log-amplitude, uniform
    dead_probability: float = Field(default=0.15, ge=0, le=1)
    dead_fraction: Range = (0.0, 0.15)
    polarity_probability: float = Field(default=0.1, ge=0, le=1)
    polarity_fraction: Range = (0.0, 0.1)
    statics_probability: float = Field(default=0.3, ge=0, le=1)
    statics: Range = (0.0, 0.05)  # delay std times the image's top frequency (cycles), uniform
    air_probability: float = Field(default=0.25, ge=0, le=1)  # active records only
    air_velocity: Range = (330.0, 345.0)  # m/s, uniform
    air_ratio: Range = (0.05, 1.0)  # over the surface waves' rms, log-uniform
    body_probability: float = Field(default=0.35, ge=0, le=1)
    body_count: tuple[int, int] = (1, 3)
    body_ratio: Range = (0.03, 0.6)  # log-uniform
    hum_probability: float = Field(default=0.2, ge=0, le=1)  # mains, in phase on all traces
    hum_ratio: Range = (0.1, 20.0)  # log-uniform
    hum_width: Range = (0.2, 1.5)  # Hz, uniform
    backward_probability: float = Field(default=0.2, ge=0, le=1)  # reflected surface waves
    backward_ratio: Range = (0.05, 0.4)  # log-uniform
    stack_probability: float = Field(default=0.3, ge=0, le=1)  # images averaged, noise redrawn
    stack_count: tuple[int, int] = (2, 4)


class ScenarioPrior(_Section):
    """Images with nothing to pick, for the quality head: the rest are ordinary surveys."""

    noise_only_probability: float = Field(default=0.04, ge=0, le=1)
    coherent_only_probability: float = Field(default=0.02, ge=0, le=1)
    off_range_probability: float = Field(default=0.03, ge=0, le=1)  # axes that miss M0


class LabelConfig(_Section):
    """When M0 counts as pickable at a frequency, and an image as pickable."""

    tolerance: float = Field(default=0.05, gt=0)  # the ridge's peak within 5 % of M0
    margin: float = Field(default=0.1, ge=0, lt=1)  # above the floor, share of floor-to-ceiling
    # ... and above what random phases reach with this probability (the Rayleigh tail).
    false_alarm: float = Field(default=0.05, gt=0, lt=1)
    relative_height: float = Field(default=0.3, ge=0, le=1)  # of the column's highest peak
    alias_wavelength: float = Field(default=2.0, ge=0)  # shortest wavelength, in spacings
    min_run: int = Field(default=3, ge=1)  # columns, for a visible stretch to count
    min_run_share: float = Field(default=0.01, ge=0, le=1)  # ... or this share of them
    gap: int = Field(default=1, ge=0)  # columns bridged between stretches
    min_octaves: float = Field(default=0.5, ge=0)  # wavelength span of a pickable image
    quality_octaves: float = Field(default=3.0, gt=0)  # the span that rates 1


class SynthesisConfig(_Section):
    array: ArrayPrior = Field(default_factory=ArrayPrior)
    scale: ScalePrior = Field(default_factory=ScalePrior)
    axes: AxesPrior = Field(default_factory=AxesPrior)
    wavefield: WavefieldPrior = Field(default_factory=WavefieldPrior)
    noise: NoisePrior = Field(default_factory=NoisePrior)
    scenarios: ScenarioPrior = Field(default_factory=ScenarioPrior)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    dense_frequencies: int = Field(default=512, ge=64)  # samples of the true curves kept
