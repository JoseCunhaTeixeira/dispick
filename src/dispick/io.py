"""Picking image files as sigpipe and PAC save them, without sigpipe.

sigpipe writes a dispersion image as HDF5 (`fv_map`, `fs`, `vs`, `type`, `source`,
`receivers`, `acquisition_kind`); its curves as a CSV of blocks (a header, then frequency,
velocity and uncertainty per line, blocks separated by `---`). `pick_files` reads the one and
writes the other, so PAC opens dispick's picks like its own, plus a JSON of dispick's verdict.
"""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from dispick.features import Geometry, geometry_from_receivers
from dispick.inference.picker import ImageInput, Picker, PickSettings
from dispick.inference.result import PickResult


@dataclass(frozen=True, slots=True)
class ImageFile:
    path: Path
    fv_map: np.ndarray
    frequencies: np.ndarray
    velocities: np.ndarray
    velocity_type: str
    source: tuple[float, float, float] | None
    receivers: np.ndarray | None  # (N, 3)
    acquisition_kind: str

    @property
    def geometry(self) -> Geometry | None:
        if self.receivers is None or self.receivers.ndim != 2 or self.receivers.shape[1] != 3:
            return None
        return geometry_from_receivers(self.receivers[:, 0], self.receivers[:, 2])


def _array(file: h5py.File, key: str, dtype: type = np.float64) -> np.ndarray | None:
    obj = file.get(key)
    return np.asarray(obj[()], dtype=dtype) if isinstance(obj, h5py.Dataset) else None


def _text(file: h5py.File, key: str) -> str:
    obj = file.get(key)
    if not isinstance(obj, h5py.Dataset):
        return ""
    value = obj[()]
    return value.decode() if isinstance(value, bytes) else str(value)


def read_image(path: Path) -> ImageFile:
    with h5py.File(path, "r") as file:
        fv_map = _array(file, "fv_map", np.float32)
        frequencies = _array(file, "fs")
        velocities = _array(file, "vs")
        if fv_map is None or frequencies is None or velocities is None:
            raise ValueError(f"{path}: not a sigpipe dispersion image (fv_map, fs, vs)")
        source = _array(file, "source")
        return ImageFile(
            path=Path(path),
            fv_map=fv_map,
            frequencies=frequencies,
            velocities=velocities,
            velocity_type=_text(file, "type"),
            source=(
                (float(source[0]), float(source[1]), float(source[2]))
                if source is not None and source.size == 3
                else None
            ),
            receivers=_array(file, "receivers"),
            acquisition_kind=_text(file, "acquisition_kind"),
        )


def lorentzian_uncertainty(
    fs: np.ndarray, vs: np.ndarray, geometry: Geometry, a: float = 0.5
) -> np.ndarray:
    """sigpipe's (and PAC's) velocity uncertainty from the array's resolving power, between
    5 m/s and 40 % of the velocity."""
    n, dx = geometry.n_receivers, geometry.spacing
    factor = 10 ** (1 / np.sqrt(n * dx))
    left = 1 / (1 / vs + 1e-12 - 1 / (2 * fs * n * factor * dx + 1e-12))
    right = 1 / (1 / vs + 1e-12 + 1 / (2 * fs * n * factor * dx + 1e-12))
    raw = 10**-a * np.abs(left - right)
    out = np.where(raw > 0.4 * vs, 0.4 * vs, raw)
    return np.where(raw < 5, 5, out).astype(np.float32)


def write_curve_csv(image: ImageFile, result: PickResult, path: Path, label: str = "M") -> Path:
    """The picked M0 in sigpipe's curve format (mode (label, 0), Lorentzian uncertainties when
    the array is known, the network's otherwise)."""
    fs, vs, errors = result.curve
    geometry = image.geometry
    if geometry is not None:
        errors = lorentzian_uncertainty(fs, vs, geometry)
    source = list(image.source) if image.source is not None else [float("nan")] * 3
    receivers = image.receivers.tolist() if image.receivers is not None else []
    lines = [
        f"type: {image.velocity_type}",
        f"mode: {(label, 0)!r}",
        f"acquisition_kind: {image.acquisition_kind}",
        f"source: {json.dumps(source)}",
        f"receivers: {json.dumps(receivers)}",
        "frequency_Hz,phase_velocity_m/s,velocity_std_m/s",
        *(f"{f:.6f},{v:.6f},{e:.6f}" for f, v, e in zip(fs, vs, errors, strict=True)),
    ]
    path.write_text("\n".join(lines) + "\n\n---\n\n")
    return path


def read_curves_csv(path: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The curves of a sigpipe curve file: (mode label, frequencies, velocities) per block."""
    curves = []
    for block in path.read_text().split("---"):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        label, rows, started = "", [], False
        for line in lines:
            if line.startswith("mode:"):
                wave, _, number = line.removeprefix("mode:").strip(" ()").partition(",")
                label = wave.strip(" '\"") + number.strip()
            elif line.startswith("frequency_Hz"):
                started = True
            elif started:
                rows.append([float(x) for x in line.split(",")[:2]])
        if rows:
            table = np.asarray(rows)
            curves.append((label, table[:, 0], table[:, 1]))
    return curves


def existing_m0(image_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """The M0 curve already picked next to an image (PAC/PACo's DispersionCurves_0000.csv
    in the window's folder), if any."""
    path = image_path.with_name(image_path.name.replace("DispersionImage", "DispersionCurves"))
    path = path.with_suffix(".csv")
    if not path.exists():
        return None
    for label, fs, vs in read_curves_csv(path):
        if label in ("M0", "R0"):
            return fs, vs
    return None


def _name(path: Path) -> str:
    # PAC keeps one image per window folder: xmid_<x>/DispersionImage_0000.hdf5.
    return f"{path.parent.name}_{path.stem}" if path.parent.name else path.stem


def pick_files(
    paths: Sequence[Path],
    out: Path,
    model: Path | None = None,
    threshold: float = 0.5,
    figure: bool = False,
) -> Iterator[str]:
    """Pick each image file; write <name>_M0.csv (if anything is picked) and <name>.json.
    Yields a line per file."""
    out.mkdir(parents=True, exist_ok=True)
    picker = Picker.load(model)
    images = [read_image(path) for path in paths]
    results = picker.pick_all(
        [ImageInput(i.fv_map, i.frequencies, i.velocities, i.geometry) for i in images],
        PickSettings(threshold=threshold),
    )
    for image, result in zip(images, results, strict=True):
        name = _name(image.path)
        record = {
            "image": str(image.path),
            "geometry_known": image.geometry is not None,
        } | result.to_dict()
        (out / f"{name}.json").write_text(json.dumps(record, indent=2))
        if result.n_picked >= 2:
            write_curve_csv(image, result, out / f"{name}_M0.csv")
        span = result.wavelength_range
        spans = f", wavelengths {span[0]:.2f}-{span[1]:.2f} m" if span else ""
        yield (
            f"{image.path}: {result.image.verdict} (p={result.image.pickable:.2f}, "
            f"quality {result.image.quality:.2f}), {result.n_picked} points{spans}"
        )
    if figure:
        from dispick.evaluation.figures import plot_picks

        plot_picks(
            [(i.fv_map, i.frequencies, i.velocities) for i in images],
            results,
            out / "picks.png",
            titles=[_name(i.path) for i in images],
            references=[existing_m0(i.path) for i in images],
        )
