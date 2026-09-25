"""From the network's outputs on the canonical grid to one velocity per frequency.

The network gives, per frequency, a distribution over velocity bins and the probability that
M0 can be picked there. Where one column hesitates between two ridges, its neighbours settle
it: the path is decoded jointly over frequency (Viterbi), each column's log-probability weighed
by its presence, jumps between neighbouring columns costing `jump_cost` per bin. The chosen
bin is then refined to a fraction of a bin (the mean over its neighbours), and its spread gives
the uncertainty.
"""

import numpy as np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def viterbi(log_probs: np.ndarray, weights: np.ndarray, jump_cost: float) -> np.ndarray:
    """The best path (n_f,) of bins through (n_f, n_v) log-probabilities, each column's
    weighed by `weights`, a jump of d bins between neighbouring columns costing jump_cost x d."""
    n_f, n_v = log_probs.shape
    bins = np.arange(n_v)
    transition = -jump_cost * np.abs(bins[:, None] - bins[None, :]).astype(np.float64)
    score = weights[0] * log_probs[0]
    back = np.zeros((n_f, n_v), dtype=np.int64)
    for f in range(1, n_f):
        candidates = score[None, :] + transition  # (to, from)
        back[f] = np.argmax(candidates, axis=1)
        score = candidates[bins, back[f]] + weights[f] * log_probs[f]
    path = np.empty(n_f, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for f in range(n_f - 1, 0, -1):
        path[f - 1] = back[f, path[f]]
    return path


def refine(probabilities: np.ndarray, path: np.ndarray, window: int = 3) -> np.ndarray:
    """Fractional bins: the probability-weighted mean over +-`window` bins around the path."""
    n_v = probabilities.shape[1]
    offsets = np.arange(-window, window + 1)
    index = np.clip(path[:, None] + offsets[None, :], 0, n_v - 1)
    local = np.take_along_axis(probabilities, index, axis=1)
    return np.sum(local * index, axis=1) / np.maximum(np.sum(local, axis=1), 1e-12)


def spread(probabilities: np.ndarray, centre: np.ndarray, window: int) -> np.ndarray:
    """Standard deviation (bins) of each column's distribution within +-`window` bins of
    `centre`: how sharply the network places the ridge."""
    n_v = probabilities.shape[1]
    bins = np.arange(n_v)[None, :]
    inside = np.abs(bins - centre[:, None]) <= window
    local = np.where(inside, probabilities, 0.0)
    total = np.maximum(local.sum(axis=1), 1e-12)
    variance = np.sum(local * (bins - centre[:, None]) ** 2, axis=1) / total
    return np.sqrt(variance)


def decode(
    logits: np.ndarray,
    presence_logits: np.ndarray,
    method: str = "viterbi",
    jump_cost: float = 0.1,
    spread_window: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fractional bins, their spread (bins) and the presence probability, per canonical
    frequency, from one mode's (n_f, n_v) logits and (n_f,) presence logits."""
    probabilities = softmax(logits.astype(np.float64), axis=1)
    presence = sigmoid(presence_logits.astype(np.float64))
    if method == "viterbi":
        path = viterbi(np.log(np.maximum(probabilities, 1e-30)), presence, jump_cost)
    elif method == "argmax":
        path = np.argmax(probabilities, axis=1)
    else:
        raise ValueError(f"unknown decoding method {method!r}: 'viterbi' or 'argmax'")
    bins = refine(probabilities, path)
    return bins, spread(probabilities, bins, spread_window), presence


def longest_run(
    mask: np.ndarray,
    frequencies: np.ndarray,
    max_gap: int = 1,
    velocities: np.ndarray | None = None,
) -> np.ndarray:
    """The mask reduced to its widest run, gaps of up to `max_gap` columns bridged (the bridged
    columns stay unpicked). Widest in octaves of wavelength when `velocities` are given (what a
    curve is worth to an inversion), else in octaves of frequency; the mask's points must have
    positive frequencies (and velocities)."""
    indices = np.flatnonzero(mask)
    out = np.zeros_like(mask, dtype=bool)
    if indices.size == 0:
        return out
    breaks = np.flatnonzero(np.diff(indices) > max_gap + 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [indices.size - 1]])
    lows, highs = indices[starts], indices[ends]
    if velocities is not None:
        widths = np.abs(
            np.log(velocities[lows] / frequencies[lows])
            - np.log(velocities[highs] / frequencies[highs])
        )
    else:
        widths = np.log(frequencies[highs] / frequencies[lows])
    best = int(np.argmax(widths))
    out[lows[best] : highs[best] + 1] = mask[lows[best] : highs[best] + 1]
    return out
