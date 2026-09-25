"""Picking M0 on phase-shift dispersion images, whatever their axes and array.

    picker = Picker.load()                       # DISPICK_MODEL, or the packaged model
    result = picker.pick(fv_map, fs, vs, Geometry(n_receivers=24, spacing=0.25))
    result.image.verdict                         # "pickable", "doubtful" or "unpickable"
    fs_m0, vs_m0, err_m0 = result.curve          # the picked points

An image is resampled onto the network's grid, over its own axes; the network's outputs are
decoded there and brought back to the image's own frequencies. With `zoom`, a second pass
looks again at the band the first pass picked, on a grid that spans only it (and up to twice
its top velocity): finer bins where the curve was a thin line in a wide image.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dispick.features import Geometry, input_channels
from dispick.grid import CanonicalAxes, CanonicalGrid, check_axis, to_canonical
from dispick.inference.backends import Backend, ModelCard, load_backend
from dispick.inference.decode import decode, longest_run, sigmoid
from dispick.inference.result import ImageAssessment, PickResult

MODEL_ENVIRONMENT = "DISPICK_MODEL"
PACKAGED_MODEL = Path(__file__).resolve().parents[1] / "models" / "dispick.onnx"


def default_model_path() -> Path:
    """$DISPICK_MODEL, else the model packaged with dispick."""
    if value := os.environ.get(MODEL_ENVIRONMENT):
        return Path(value)
    if PACKAGED_MODEL.exists():
        return PACKAGED_MODEL
    raise FileNotFoundError(
        f"no dispick model: pass one, set ${MODEL_ENVIRONMENT} to an .onnx or .pt file, or "
        f"place one at {PACKAGED_MODEL}"
    )


@dataclass(frozen=True, slots=True)
class PickSettings:
    threshold: float = 0.5  # presence a point needs to be picked
    continuous: bool = True  # keep only the widest continuous stretch
    max_gap: int = 2  # columns of the image a stretch bridges
    method: str = "viterbi"  # or "argmax"
    jump_cost: float = 0.1  # per canonical velocity bin, between neighbouring frequencies
    zoom: bool = False  # a second pass on the picked band
    batch_size: int = 16


@dataclass(frozen=True, slots=True)
class ImageInput:
    """One phase-shift image as sigpipe stores it: fv_map[n_f, n_v] over increasing axes."""

    fv_map: np.ndarray
    frequencies: np.ndarray
    velocities: np.ndarray
    geometry: Geometry | None = None


@dataclass(frozen=True, slots=True)
class _Canonical:
    inputs: np.ndarray
    axes: CanonicalAxes


class Picker:
    def __init__(self, backend: Backend, card: ModelCard) -> None:
        self.backend = backend
        self.card = card
        self.grid = CanonicalGrid(*card.grid)

    @classmethod
    def load(cls, path: Path | str | None = None, device: str = "cpu") -> Picker:
        """The model at `path` (.onnx with its .json card, or a .pt checkpoint)."""
        backend, card = load_backend(Path(path) if path else default_model_path(), device)
        return cls(backend, card)

    def pick(
        self,
        fv_map: np.ndarray,
        frequencies: np.ndarray,
        velocities: np.ndarray,
        geometry: Geometry | None = None,
        settings: PickSettings | None = None,
    ) -> PickResult:
        return self.pick_all([ImageInput(fv_map, frequencies, velocities, geometry)], settings)[0]

    def pick_all(
        self, images: Sequence[ImageInput], settings: PickSettings | None = None
    ) -> list[PickResult]:
        """Pick several images, batched through the network."""
        settings = settings or PickSettings()
        images = [_checked(image) for image in images]
        first = self._run([self._canonical(image, None, None) for image in images], settings)
        results = [
            self._result(image, [pass_], settings)
            for image, pass_ in zip(images, first, strict=True)
        ]
        if not settings.zoom:
            return results
        zooms = [_zoom_ranges(image, result) for image, result in zip(images, results, strict=True)]
        todo = [i for i, ranges in enumerate(zooms) if ranges is not None]
        if not todo:
            return results
        second = self._run(
            [self._canonical(images[i], *zooms[i]) for i in todo],  # pyright: ignore[reportCallIssue]
            settings,
        )
        for i, pass_ in zip(todo, second, strict=True):
            results[i] = self._result(images[i], [first[i], pass_], settings)
        return results

    def assess(
        self,
        fv_map: np.ndarray,
        frequencies: np.ndarray,
        velocities: np.ndarray,
        geometry: Geometry | None = None,
    ) -> ImageAssessment:
        """The image verdict alone (the same network pass as `pick`)."""
        return self.pick(fv_map, frequencies, velocities, geometry).image

    def _canonical(
        self,
        image: ImageInput,
        f_range: tuple[float, float] | None,
        v_range: tuple[float, float] | None,
    ) -> _Canonical:
        canonical, axes = to_canonical(
            image.fv_map, image.frequencies, image.velocities, self.grid, f_range, v_range
        )
        return _Canonical(input_channels(canonical, axes, image.geometry), axes)

    def _run(self, items: list[_Canonical], settings: PickSettings) -> list[_Pass]:
        passes: list[_Pass] = []
        for start in range(0, len(items), settings.batch_size):
            chunk = items[start : start + settings.batch_size]
            outputs = self.backend.run(np.stack([item.inputs for item in chunk]))
            for k, item in enumerate(chunk):
                bins, spread, presence = decode(
                    outputs["logits"][k, 0],
                    outputs["presence"][k, 0],
                    method=settings.method,
                    jump_cost=settings.jump_cost,
                )
                step = (item.axes.v_range[1] - item.axes.v_range[0]) / (self.grid.n_velocities - 1)
                image = sigmoid(outputs["image"][k].astype(np.float64))
                passes.append(
                    _Pass(
                        axes=item.axes,
                        velocities=item.axes.velocity_of_bin(bins),
                        uncertainties=spread * step,
                        presence=presence,
                        image=ImageAssessment(
                            pickable=float(image[0]),
                            quality=float(image[1]),
                            higher_mode_share=float(image[2]),
                        ),
                    )
                )
        return passes

    def _result(self, image: ImageInput, passes: list[_Pass], settings: PickSettings) -> PickResult:
        """The passes at the image's frequencies, each later pass overriding the earlier ones
        within its band; the verdict is the first (whole image) pass's."""
        fs = image.frequencies
        velocities = np.full(fs.size, np.nan)
        uncertainties = np.full(fs.size, np.nan)
        presence = np.zeros(fs.size)
        for pass_ in passes:
            low, high = pass_.axes.f_range
            inside = (fs >= low) & (fs <= high)
            grid_f = pass_.axes.frequencies
            velocities[inside] = np.interp(fs[inside], grid_f, pass_.velocities)
            uncertainties[inside] = np.interp(fs[inside], grid_f, pass_.uncertainties)
            presence[inside] = np.interp(fs[inside], grid_f, pass_.presence)
        v_low, v_high = float(image.velocities[0]), float(image.velocities[-1])
        picked = (
            (presence >= settings.threshold)
            & (fs > 0)
            & np.isfinite(velocities)
            & (velocities >= v_low)
            & (velocities <= v_high)
        )
        if settings.continuous:
            picked = longest_run(picked, fs, settings.max_gap)
        return PickResult(
            frequencies=fs,
            velocities=velocities,
            uncertainties=uncertainties,
            presence=presence,
            picked=picked,
            image=passes[0].image,
            threshold=settings.threshold,
        )


@dataclass(frozen=True, slots=True)
class _Pass:
    axes: CanonicalAxes
    velocities: np.ndarray
    uncertainties: np.ndarray
    presence: np.ndarray
    image: ImageAssessment


def _checked(image: ImageInput) -> ImageInput:
    frequencies = check_axis(image.frequencies, "frequencies")
    velocities = check_axis(image.velocities, "velocities")
    fv_map = np.asarray(image.fv_map, dtype=np.float32)
    if fv_map.shape != (frequencies.size, velocities.size):
        raise ValueError(
            f"fv_map shape {fv_map.shape} does not match the axes "
            f"({frequencies.size}, {velocities.size})"
        )
    if not np.isfinite(fv_map).all():
        fv_map = np.nan_to_num(fv_map, nan=0.0, posinf=1.0, neginf=0.0)
    return ImageInput(fv_map, frequencies, velocities, image.geometry)


def _zoom_ranges(
    image: ImageInput, result: PickResult
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """The sub-ranges a second pass should span, or None when the first pass picked too
    little or already used most of the grid."""
    f, v, e = result.curve
    if f.size < 3:
        return None
    f0, f1 = float(image.frequencies[0]), float(image.frequencies[-1])
    v0, v1 = float(image.velocities[0]), float(image.velocities[-1])
    band = float(f[-1] - f[0])
    f_low = max(f0, float(f[0]) - 0.25 * band)
    f_high = min(f1, float(f[-1]) + 0.25 * band)
    v_high = min(v1, 2.0 * float(np.max(v + e)))
    if (f_high - f_low) > 0.7 * (f1 - f0) and (v_high - v0) > 0.7 * (v1 - v0):
        return None
    # At least a few of the image's columns and velocities, for the grid to span.
    if np.sum((image.frequencies >= f_low) & (image.frequencies <= f_high)) < 4:
        return None
    if np.sum((image.velocities >= v0) & (image.velocities <= v_high)) < 16:
        return None
    return (f_low, f_high), (v0, v_high)
