"""Random receiver arrays: active shots at a distance, or passive virtual shots on a receiver."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from dispick.features import Geometry
from dispick.physics.earth import log_uniform
from dispick.synthesis.config import ArrayPrior

type Kind = Literal["active", "passive"]


@dataclass(frozen=True, slots=True)
class Array:
    """A receiver array and its source.

    `offsets` are the nominal source-receiver distances the image is computed with (from the
    survey's files); `true_offsets` are where the receivers really are, which the wavefield
    follows. A passive virtual shot sits on the first receiver: its offset is 0, and the phase
    shift weighs it out (sqrt(offset))."""

    kind: Kind
    offsets: np.ndarray
    true_offsets: np.ndarray
    geometry: Geometry


def sample_array(prior: ArrayPrior, rng: np.random.Generator) -> Array:
    weights = np.array([weight for _, _, weight in prior.n_receivers], dtype=float)
    low, high, _ = prior.n_receivers[int(rng.choice(weights.size, p=weights / weights.sum()))]
    n = int(rng.integers(low, high + 1))
    spacing = log_uniform(rng, prior.spacing)
    steps = np.full(n - 1, spacing)
    if rng.random() < prior.irregular_probability:
        steps *= rng.uniform(0.7, 1.3, n - 1)
    positions = np.concatenate([[0.0], np.cumsum(steps)])
    kind: Kind = "active" if rng.random() < prior.active_probability else "passive"
    if kind == "active":
        offsets = positions + positions[-1] * log_uniform(rng, prior.near_offset)
    else:
        offsets = positions
    true_offsets = offsets.copy()
    if rng.random() < prior.jitter_probability:
        error = rng.normal(0.0, rng.uniform(*prior.jitter) * spacing, n)
        if kind == "passive":
            error[0] = 0.0  # the virtual source is the receiver itself
        true_offsets = np.maximum(offsets + error, 0.0)
    return Array(
        kind=kind,
        offsets=offsets,
        true_offsets=true_offsets,
        geometry=Geometry(n_receivers=n, spacing=float(np.median(steps))),
    )
