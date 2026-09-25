"""Random layered earth models of the near surface: the ground truth behind the synthetic images.

Each family is a kind of ground a MASW survey meets, and each gives the network a different
picture: steady dispersion, a ridge that flattens early, higher modes stealing the energy at high
frequencies, osculating modes. The half-space is always the stiffest layer, so that every mode
disba computes is a guided (non-leaky) mode.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

type Range = tuple[float, float]


class Family(StrEnum):
    NORMAL = "normal"  # velocity increasing with depth, by steps
    GRADIENT = "gradient"  # a power-law increase (confining pressure), over bedrock or not
    STIFF_TOP = "stiff_top"  # a stiff crust over softer ground: inverse dispersion at high f
    LVL = "lvl"  # a low-velocity layer at depth
    BEDROCK = "bedrock"  # soft cover over much stiffer bedrock: strong higher modes
    RANDOM = "random"  # a random walk in log(Vs)


FAMILIES: tuple[Family, ...] = tuple(Family)


@dataclass(frozen=True, slots=True)
class LayeredModel:
    """A 1D elastic model in SI units (m, m/s, kg/m³). The last layer is the half-space: its
    thickness is kept for the record, never used."""

    thickness: np.ndarray
    vs: np.ndarray
    vp: np.ndarray
    rho: np.ndarray
    family: Family

    def __post_init__(self) -> None:
        arrays = {}
        for name in ("thickness", "vs", "vp", "rho"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.ndim != 1:
                raise ValueError(f"{name} must be 1D, got shape {value.shape}")
            arrays[name] = value
        sizes = {value.size for value in arrays.values()}
        if len(sizes) != 1:
            raise ValueError(f"thickness, vs, vp and rho must have the same length, got {sizes}")
        if arrays["vs"].size < 2:
            raise ValueError("a model needs at least one layer over the half-space")
        if np.any(arrays["thickness"][:-1] <= 0):
            raise ValueError("layer thicknesses must be > 0")
        if np.any(arrays["vs"] <= 0) or np.any(arrays["rho"] <= 0):
            raise ValueError("vs and rho must be > 0")
        # Poisson's ratio above ~0.05: the Lamé parameter lambda stays positive.
        if np.any(arrays["vp"] < 1.45 * (1 - 1e-6) * arrays["vs"]):
            raise ValueError("vp must be at least 1.45 x vs")
        for name, value in arrays.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "family", Family(self.family))

    @property
    def n_layers(self) -> int:
        """Layers including the half-space."""
        return int(self.vs.size)

    @property
    def depth(self) -> float:
        """Depth to the top of the half-space, m."""
        return float(np.sum(self.thickness[:-1]))

    @property
    def tops(self) -> np.ndarray:
        """Depth of the top of each layer, m."""
        return np.concatenate([[0.0], np.cumsum(self.thickness[:-1])])


class EarthPrior(BaseModel):
    """Distributions the models are drawn from. Ranges are sampled log-uniformly unless said."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    family_weights: dict[Family, float] = Field(
        default_factory=lambda: {
            Family.NORMAL: 0.30,
            Family.GRADIENT: 0.20,
            Family.STIFF_TOP: 0.15,
            Family.LVL: 0.10,
            Family.BEDROCK: 0.15,
            Family.RANDOM: 0.10,
        }
    )
    vs_surface: Range = (60.0, 700.0)
    depth: Range = (1.0, 60.0)  # to the half-space
    n_layers: tuple[int, int] = (1, 6)  # layers over the half-space, uniform, both included
    thickness_growth: Range = (1.0, 2.0)  # how fast layers thicken with depth
    step_ratio: Range = (1.05, 2.2)  # Vs ratio between consecutive layers
    halfspace_ratio: Range = (1.1, 3.0)  # half-space Vs over the layer above it
    stiff_top_ratio: Range = (1.3, 3.5)  # crust Vs over the ground under it
    stiff_top_fraction: Range = (0.03, 0.35)  # crust thickness over the depth to the half-space
    lvl_ratio: Range = (0.45, 0.85)  # low-velocity layer Vs over the layer above it
    bedrock_ratio: Range = (2.5, 10.0)  # bedrock Vs over the cover
    gradient_exponent: tuple[float, float] = (0.15, 0.7)  # uniform
    gradient_depth: Range = (0.2, 5.0)  # the reference depth z0 of (1 + z / z0)^p, m
    vs_max: float = Field(default=3500.0, gt=0)
    vp_vs: Range = (1.6, 2.8)  # above the water table
    vp_vs_jitter: float = Field(default=0.05, ge=0, lt=0.5)
    water_table_probability: float = Field(default=0.3, ge=0, le=1)
    water_table_depth: tuple[float, float] = (
        0.05,
        1.2,
    )  # uniform, over the depth to the half-space
    vp_water: tuple[float, float] = (1450.0, 1600.0)  # uniform
    vp_vs_max: float = Field(default=15.0, gt=1.45)

    @model_validator(mode="after")
    def _check(self) -> EarthPrior:
        if not self.family_weights or any(w < 0 for w in self.family_weights.values()):
            raise ValueError("family_weights must be non-negative, with at least one family")
        if sum(self.family_weights.values()) <= 0:
            raise ValueError("family_weights must not all be 0")
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, tuple) and len(value) == 2 and value[0] > value[1]:  # pyright: ignore[reportUnknownArgumentType]
                raise ValueError(f"{name}: the lower bound exceeds the upper one ({value})")
        return self


def log_uniform(rng: np.random.Generator, bounds: Range) -> float:
    low, high = bounds
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_model(prior: EarthPrior, rng: np.random.Generator) -> LayeredModel:
    """One random model of the prior's families."""
    families = [family for family, weight in prior.family_weights.items() if weight > 0]
    weights = np.array([prior.family_weights[family] for family in families], dtype=float)
    family = families[int(rng.choice(len(families), p=weights / weights.sum()))]
    depth = log_uniform(rng, prior.depth)
    thickness, vs = _SAMPLERS[family](prior, rng, depth)
    vs = np.minimum(vs, prior.vs_max)
    # The half-space is the stiffest layer: every mode is then guided.
    vs[-1] = max(vs[-1], 1.05 * float(np.max(vs[:-1])))
    thickness, vs, vp, rho = _elastic(prior, rng, thickness, vs)
    return LayeredModel(thickness=thickness, vs=vs, vp=vp, rho=rho, family=family)


def _partition(prior: EarthPrior, rng: np.random.Generator, n: int, depth: float) -> np.ndarray:
    """n thicknesses summing to depth, thickening with depth on average."""
    growth = log_uniform(rng, prior.thickness_growth)
    weights = np.exp(rng.normal(0.0, 0.6, n)) * growth ** np.arange(n)
    return depth * weights / weights.sum()


def _n_layers(prior: EarthPrior, rng: np.random.Generator, minimum: int = 1) -> int:
    low, high = prior.n_layers
    return int(rng.integers(max(low, minimum), max(high, minimum) + 1))


def _steps(prior: EarthPrior, rng: np.random.Generator, start: float, n: int) -> np.ndarray:
    ratios = np.array([log_uniform(rng, prior.step_ratio) for _ in range(n - 1)])
    return start * np.concatenate([[1.0], np.cumprod(ratios)])


def _with_halfspace(
    prior: EarthPrior, rng: np.random.Generator, thickness: np.ndarray, vs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    halfspace = vs[-1] * log_uniform(rng, prior.halfspace_ratio)
    return np.append(thickness, thickness[-1]), np.append(vs, halfspace)


def _normal(
    prior: EarthPrior, rng: np.random.Generator, depth: float
) -> tuple[np.ndarray, np.ndarray]:
    n = _n_layers(prior, rng)
    vs = _steps(prior, rng, log_uniform(rng, prior.vs_surface), n)
    return _with_halfspace(prior, rng, _partition(prior, rng, n, depth), vs)


def _gradient(
    prior: EarthPrior, rng: np.random.Generator, depth: float
) -> tuple[np.ndarray, np.ndarray]:
    n = int(rng.integers(8, 17))
    # Thin sublayers at the top, where the gradient is steepest.
    thickness = depth * np.geomspace(1.0, 4.0, n) / np.sum(np.geomspace(1.0, 4.0, n))
    middles = np.cumsum(thickness) - thickness / 2
    exponent = rng.uniform(*prior.gradient_exponent)
    z0 = log_uniform(rng, prior.gradient_depth)
    vs = log_uniform(rng, prior.vs_surface) * (1 + middles / z0) ** exponent
    if rng.random() < 0.5:
        return np.append(thickness, thickness[-1]), np.append(
            vs, vs[-1] * log_uniform(rng, prior.bedrock_ratio)
        )
    return np.append(thickness, thickness[-1]), np.append(vs, vs[-1] * rng.uniform(1.02, 1.2))


def _stiff_top(
    prior: EarthPrior, rng: np.random.Generator, depth: float
) -> tuple[np.ndarray, np.ndarray]:
    crust = depth * log_uniform(rng, prior.stiff_top_fraction)
    n = _n_layers(prior, rng)
    soft = log_uniform(rng, prior.vs_surface)
    vs_below = _steps(prior, rng, soft, n)
    thickness = np.concatenate([[crust], _partition(prior, rng, n, depth - crust)])
    vs = np.concatenate([[soft * log_uniform(rng, prior.stiff_top_ratio)], vs_below])
    halfspace = max(vs.max(), vs_below[-1] * log_uniform(rng, prior.halfspace_ratio))
    return np.append(thickness, thickness[-1]), np.append(vs, halfspace)


def _lvl(
    prior: EarthPrior, rng: np.random.Generator, depth: float
) -> tuple[np.ndarray, np.ndarray]:
    n = _n_layers(prior, rng, minimum=3)
    vs = _steps(prior, rng, log_uniform(rng, prior.vs_surface), n)
    k = int(rng.integers(1, n - 1))
    vs[k] = vs[k - 1] * log_uniform(rng, prior.lvl_ratio)
    return _with_halfspace(prior, rng, _partition(prior, rng, n, depth), vs)


def _bedrock(
    prior: EarthPrior, rng: np.random.Generator, depth: float
) -> tuple[np.ndarray, np.ndarray]:
    n = int(rng.integers(1, 4))
    cover = log_uniform(rng, prior.vs_surface) * np.cumprod(
        np.concatenate([[1.0], rng.uniform(1.0, 1.4, n - 1)])
    )
    thickness = _partition(prior, rng, n, depth)
    return np.append(thickness, thickness[-1]), np.append(
        cover, cover[-1] * log_uniform(rng, prior.bedrock_ratio)
    )


def _random(
    prior: EarthPrior, rng: np.random.Generator, depth: float
) -> tuple[np.ndarray, np.ndarray]:
    n = _n_layers(prior, rng, minimum=2)
    surface = log_uniform(rng, prior.vs_surface)
    steps = np.clip(rng.normal(0.2, 0.35, n - 1), -0.7, 1.0)
    vs = np.clip(surface * np.exp(np.concatenate([[0.0], np.cumsum(steps)])), 0.5 * surface, None)
    return _with_halfspace(prior, rng, _partition(prior, rng, n, depth), vs)


_SAMPLERS = {
    Family.NORMAL: _normal,
    Family.GRADIENT: _gradient,
    Family.STIFF_TOP: _stiff_top,
    Family.LVL: _lvl,
    Family.BEDROCK: _bedrock,
    Family.RANDOM: _random,
}


def _elastic(
    prior: EarthPrior, rng: np.random.Generator, thickness: np.ndarray, vs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vp and density for the Vs profile, with an optional water table: below it, Vp is at
    least the water's."""
    ratio = log_uniform(rng, prior.vp_vs) * np.exp(rng.normal(0.0, prior.vp_vs_jitter, vs.size))
    vp = vs * np.maximum(ratio, 1.5)
    saturated = np.zeros(vs.size, dtype=bool)
    if rng.random() < prior.water_table_probability:
        table = rng.uniform(*prior.water_table_depth) * float(np.sum(thickness[:-1]))
        thickness, vs, vp, first = _split_at(thickness, vs, vp, table)
        saturated = np.arange(vs.size) >= first
        vp = np.where(saturated, np.maximum(vp, rng.uniform(*prior.vp_water)), vp)
    vp = np.clip(vp, 1.45 * vs, prior.vp_vs_max * vs)
    rho = 1500.0 + 400.0 * np.clip(np.log10(vs / 80.0) / np.log10(20.0), 0.0, 1.0)
    rho = rho + rng.normal(0.0, 60.0, vs.size) + np.where(saturated, 150.0, 0.0)
    return thickness, vs, vp, np.clip(rho, 1400.0, 2700.0)


def _split_at(
    thickness: np.ndarray, vs: np.ndarray, vp: np.ndarray, depth: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """The model with an interface at `depth` (none added within 5 % of a layer's edge), and
    the index of the first layer below it."""
    tops = np.concatenate([[0.0], np.cumsum(thickness[:-1])])
    layer = int(np.searchsorted(tops, depth, side="right")) - 1
    offset = depth - tops[layer]
    if layer == vs.size - 1:
        # In the half-space: a finite layer of its properties, then the half-space.
        if offset <= 0:
            return thickness, vs, vp, layer
        return (
            np.insert(thickness, layer, offset),
            np.insert(vs, layer, vs[layer]),
            np.insert(vp, layer, vp[layer]),
            layer + 1,
        )
    if offset < 0.05 * thickness[layer]:
        return thickness, vs, vp, layer
    if thickness[layer] - offset < 0.05 * thickness[layer]:
        return thickness, vs, vp, layer + 1
    thickness = np.insert(thickness, layer + 1, thickness[layer] - offset)
    thickness[layer] = offset
    return thickness, np.insert(vs, layer, vs[layer]), np.insert(vp, layer, vp[layer]), layer + 1
