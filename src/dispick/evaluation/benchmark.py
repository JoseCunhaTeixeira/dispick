"""The picker against the synthetic truth, end to end from images at their own axes.

For every image of a benchmark: the points picked, how many lie within 5 % of M0 (`good`),
how many sit on a higher mode instead (`on_higher`), how many of the pickable points were found
(`found`); and the image verdict against its label. Totals are pooled over points, then broken
down by array size, record kind, scenario, ground family, signal-to-noise and frequency count,
where pickers differ. Baselines are scored the same way.
"""

import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from dispick.inference.picker import ImageInput, Picker, PickSettings
from dispick.synthesis.sample import SyntheticSample

TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class Pick:
    """A picker's answer for one image, at the image's frequencies."""

    velocities: np.ndarray
    picked: np.ndarray
    pickable: float | None = None  # the image verdict, for pickers that give one
    quality: float | None = None
    higher_mode_share: float | None = None


type Method = Callable[[list[SyntheticSample]], list[Pick]]


@dataclass(frozen=True, slots=True)
class ImageScore:
    visible: int
    picked: int
    good: int
    found: int
    on_higher: int
    visible_good: int  # visible points whose estimate is within 5 % (picked or not)
    error_sum: float  # relative error over picked points with a truth
    pickable_label: bool
    pickable: float | None
    quality_label: float
    quality: float | None
    share_label: float
    share: float | None
    strata: dict[str, str]


def score(sample: SyntheticSample, pick: Pick) -> ImageScore:
    truth = sample.curves[0]
    finite = np.isfinite(truth) & (truth > 0) & np.isfinite(pick.velocities)
    error = np.full(truth.shape, np.inf)
    error[finite] = np.abs(pick.velocities[finite] - truth[finite]) / truth[finite]
    close = error < TOLERANCE
    higher = sample.curves[1:]
    near_higher = np.zeros(truth.shape, dtype=bool)
    for row in higher:
        ok = np.isfinite(row) & np.isfinite(pick.velocities)
        near_higher[ok] |= np.abs(pick.velocities[ok] - row[ok]) / row[ok] < TOLERANCE
    picked, visible = pick.picked.astype(bool), sample.visible.astype(bool)
    counted = picked & finite
    return ImageScore(
        visible=int(visible.sum()),
        picked=int(picked.sum()),
        good=int((picked & close).sum()),
        found=int((picked & visible & close).sum()),
        on_higher=int((picked & near_higher & ~close).sum()),
        visible_good=int((visible & close).sum()),
        error_sum=float(np.sum(np.minimum(error[counted], 10.0))),
        pickable_label=sample.labels.pickable,
        pickable=pick.pickable,
        quality_label=sample.labels.quality,
        quality=pick.quality,
        share_label=sample.labels.higher_mode_share,
        share=pick.higher_mode_share,
        strata=strata(sample),
    )


def strata(sample: SyntheticSample) -> dict[str, str]:
    info = sample.info
    n = sample.geometry.n_receivers
    snr = float(info.get("snr_peak", math.nan))
    n_f = sample.frequencies.size
    return {
        "receivers": "3-6"
        if n <= 6
        else "7-12"
        if n <= 12
        else "13-24"
        if n <= 24
        else "25-48"
        if n <= 48
        else "49+",
        "kind": str(info.get("kind", "")),
        "scenario": str(info.get("scenario", "")),
        "family": str(info.get("family", "")),
        "snr": "<5 dB"
        if snr < 5
        else "5-15 dB"
        if snr < 15
        else "15-25 dB"
        if snr < 25
        else ">=25 dB",
        "frequencies": "<16" if n_f < 16 else "16-99" if n_f < 100 else ">=100",
    }


def summarize(scores: Iterable[ImageScore]) -> dict[str, float]:
    scores = list(scores)
    total: dict[str, float] = defaultdict(float)
    for s in scores:
        for key in ("visible", "picked", "good", "found", "on_higher", "visible_good", "error_sum"):
            total[key] += getattr(s, key)

    def ratio(a: float, b: float) -> float:
        return a / b if b > 0 else math.nan

    out = {
        "images": float(len(scores)),
        "points_picked": total["picked"],
        "points_pickable": total["visible"],
        "precision": ratio(total["good"], total["picked"]),
        "recall": ratio(total["found"], total["visible"]),
        "mode_confusion": ratio(total["on_higher"], total["picked"]),
        "mean_error": ratio(total["error_sum"], total["picked"]),
        # Where M0 is pickable, how often the estimate is right, picked or not: the velocity
        # accuracy alone, whatever the presence threshold.
        "accuracy_where_pickable": ratio(total["visible_good"], total["visible"]),
    }
    verdicts = [(s.pickable, s.pickable_label) for s in scores if s.pickable is not None]
    if verdicts:
        probability = np.array([p for p, _ in verdicts], dtype=float)
        label = np.array([t for _, t in verdicts], dtype=bool)
        said = probability >= 0.5
        tp = float((said & label).sum())
        fp = float((said & ~label).sum())
        fn = float((~said & label).sum())
        out |= {
            "pickable_accuracy": float(np.mean(said == label)),
            "pickable_f1": ratio(2 * tp, 2 * tp + fp + fn),
            "pickable_auc": auc(probability, label),
            "pickable_brier": float(np.mean((probability - label) ** 2)),
            "unpickable_rejected": ratio(float((~said & ~label).sum()), float((~label).sum())),
        }
        quality = [abs(s.quality - s.quality_label) for s in scores if s.quality is not None]
        out["quality_mae"] = float(np.mean(quality)) if quality else math.nan
        share = [abs(s.share - s.share_label) for s in scores if s.share is not None]
        out["share_mae"] = float(np.mean(share)) if share else math.nan
    return out


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve, by ranks (ties averaged)."""
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if positives == 0 or negatives == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size)
    sorted_scores = scores[order]
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def report(scores: list[ImageScore]) -> dict[str, object]:
    out: dict[str, object] = {"overall": summarize(scores)}
    keys = scores[0].strata.keys() if scores else []
    for key in keys:
        groups: dict[str, list[ImageScore]] = defaultdict(list)
        for s in scores:
            groups[s.strata[key]].append(s)
        out[key] = {name: summarize(group) for name, group in sorted(groups.items())}
    return out


def picker_method(picker: Picker, settings: PickSettings | None = None) -> Method:
    def run(samples: list[SyntheticSample]) -> list[Pick]:
        results = picker.pick_all(
            [ImageInput(s.fv_map, s.frequencies, s.velocities, s.geometry) for s in samples],
            settings,
        )
        return [
            Pick(
                velocities=r.velocities,
                picked=r.picked,
                pickable=r.image.pickable,
                quality=r.image.quality,
                higher_mode_share=r.image.higher_mode_share,
            )
            for r in results
        ]

    return run


def precomputed_method(path: Path) -> Method:
    """Picks made elsewhere (another tool's picker), stored per benchmark image as `v_<index>`
    (velocity at each of the image's frequencies, NaN where none) and `k_<index>` (picked), in
    the benchmark's order."""
    arrays = np.load(path)
    cursor = 0

    def run(samples: list[SyntheticSample]) -> list[Pick]:
        nonlocal cursor
        picks = []
        for sample in samples:
            velocities = np.asarray(arrays[f"v_{cursor:06d}"], dtype=np.float64)
            if velocities.shape != sample.frequencies.shape:
                raise ValueError(f"{path}: image {cursor} does not match the benchmark")
            picks.append(
                Pick(
                    velocities=velocities, picked=np.asarray(arrays[f"k_{cursor:06d}"], dtype=bool)
                )
            )
            cursor += 1
        return picks

    return run


def evaluate(
    samples: Iterable[SyntheticSample], methods: dict[str, Method], chunk: int = 64
) -> dict[str, list[ImageScore]]:
    scores: dict[str, list[ImageScore]] = {name: [] for name in methods}
    batch: list[SyntheticSample] = []

    def flush() -> None:
        for name, method in methods.items():
            for sample, pick in zip(batch, method(batch), strict=True):
                scores[name].append(score(sample, pick))
        batch.clear()

    for sample in samples:
        batch.append(sample)
        if len(batch) == chunk:
            flush()
    if batch:
        flush()
    return scores


def write_report(scores: dict[str, list[ImageScore]], folder: Path) -> dict[str, dict[str, object]]:
    folder.mkdir(parents=True, exist_ok=True)
    reports = {name: report(values) for name, values in scores.items()}
    (folder / "report.json").write_text(json.dumps(_clean(reports), indent=2, allow_nan=False))
    with (folder / "images.jsonl").open("w") as file:
        for name, values in scores.items():
            for index, value in enumerate(values):
                file.write(
                    json.dumps(
                        _clean({"method": name, "index": index} | asdict(value)), allow_nan=False
                    )
                    + "\n"
                )
    (folder / "report.md").write_text(markdown(reports))
    return reports


def _clean(value: object) -> object:
    """The value as strict JSON: NaN and infinities as null, numpy scalars as numbers."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, list | tuple):
        return [_clean(item) for item in value]  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def markdown(reports: dict[str, dict[str, object]]) -> str:
    """Tables of the headline numbers per method, overall and per stratum."""
    columns = (
        "precision",
        "recall",
        "mode_confusion",
        "accuracy_where_pickable",
        "pickable_f1",
        "pickable_auc",
    )
    lines = ["# dispick benchmark", ""]
    names = list(reports)
    if not names:
        return "\n".join(lines)
    sections = ["overall", *[key for key in reports[names[0]] if key != "overall"]]
    for section in sections:
        lines += [f"## {section}", "", "| method | group | images | " + " | ".join(columns) + " |"]
        lines.append("|---" * (3 + len(columns)) + "|")
        for name in names:
            groups = cast(dict[str, Any], reports[name][section])
            items = [("all", groups)] if section == "overall" else list(groups.items())
            for group, values in items:
                cells = [_cell(values.get(column)) for column in columns]
                lines.append(
                    f"| {name} | {group} | {int(values['images'])} | " + " | ".join(cells) + " |"
                )
        lines.append("")
    return "\n".join(lines)


def _cell(value: object) -> str:
    if not isinstance(value, int | float) or not math.isfinite(value):
        return "-"
    return f"{float(value):.3f}"
