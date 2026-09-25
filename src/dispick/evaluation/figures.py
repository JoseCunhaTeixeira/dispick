"""Figures for eyes: images with their picks, and the truth when there is one."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from dispick.inference.result import PickResult


def plot_picks(
    images: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    results: Sequence[PickResult],
    path: Path,
    titles: Sequence[str] | None = None,
    truths: Sequence[np.ndarray | None] | None = None,
    columns: int = 4,
) -> Path:
    """A grid of (fv_map, frequencies, velocities) images with each result's picked points
    (coloured by presence) and the remaining estimate (grey), over the truth when given."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    rows = max(1, -(-n // columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.6 * rows), squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.axis("off")
            continue
        fv_map, fs, vs = images[k]
        result = results[k]
        ax.imshow(
            fv_map.T,
            origin="lower",
            aspect="auto",
            extent=(float(fs[0]), float(fs[-1]), float(vs[0]), float(vs[-1])),
            cmap="turbo",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        truth = truths[k] if truths is not None else None
        if truth is not None:
            ax.plot(fs, truth, color="white", lw=1.2, label="M0 (true)")
        rest = ~result.picked & np.isfinite(result.velocities)
        ax.plot(fs[rest], result.velocities[rest], ".", color="0.6", ms=2)
        f, v, e = result.curve
        if f.size:
            ax.errorbar(f, v, yerr=e, fmt="none", ecolor="k", elinewidth=0.6, alpha=0.6)
            ax.scatter(
                f, v, c=result.presence[result.picked], cmap="magma", vmin=0, vmax=1, s=9, zorder=3
            )
        ax.set_xlim(float(fs[0]), float(fs[-1]))
        ax.set_ylim(float(vs[0]), float(vs[-1]))
        verdict = result.image
        head = f"{verdict.verdict} p={verdict.pickable:.2f} q={verdict.quality:.2f} hm={verdict.higher_mode_share:.2f}"
        title = f"{titles[k]}\n{head}" if titles is not None else head
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=6)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=80)
    plt.close(figure)
    return path
