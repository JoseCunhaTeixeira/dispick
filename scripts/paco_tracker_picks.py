"""PACo's classical picker (ridge tracking) on a dispick benchmark, for `dispick evaluate`.

Run with PACo's interpreter (it needs paco and sigpipe, not dispick):

    ../PACo/.venv/bin/python scripts/paco_tracker_picks.py data/benchmark.h5 runs/paco_tracker.npz

then score it next to dispick:

    dispick evaluate --model models/dispick.onnx --benchmark data/benchmark.h5 --out report/ \
        --baselines --external paco_tracker=runs/paco_tracker.npz

Each benchmark image becomes a sigpipe DispersionImage over a linear array at its nominal
offsets; PACo's `pick_modes` (its default PickingParameters) gives M0's tracked frequencies,
velocities and kept points, stored per image as `v_<index>` (velocity at each of the image's
frequencies, NaN where not tracked) and `k_<index>` (kept).
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
from paco.picking import PickingParameters, pick_modes
from sigpipe.base import Coordinate, DispersionImage, LinearAcquisition, VelocityType


def main(benchmark: Path, output: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    parameters = PickingParameters()
    with h5py.File(benchmark, "r") as file:
        names = sorted(file.keys())
        for index, name in enumerate(names):
            group = file[name]
            fs = np.asarray(group["frequencies"][:])
            vs = np.asarray(group["velocities"][:])
            offsets = np.asarray(group["offsets"][:])
            image = DispersionImage(
                fv_map=np.asarray(group["fv_map"][:], dtype=np.float32),
                fs=fs,
                vs=vs,
                type=VelocityType.PHASE,
                acquisition=LinearAcquisition(
                    source=Coordinate(0.0, 0.0, 0.0),
                    receivers=tuple(Coordinate(float(x), 0.0, 0.0) for x in offsets),
                ),
            )
            velocities = np.full(fs.size, np.nan)
            kept = np.zeros(fs.size, dtype=bool)
            try:
                modes = pick_modes(image, parameters)
            except ValueError:
                modes = []
            if modes:
                m0 = modes[0]
                rows = np.searchsorted(fs.astype(np.float32), m0.frequencies.astype(np.float32))
                rows = np.clip(rows, 0, fs.size - 1)
                velocities[rows] = m0.velocities
                kept[rows] = m0.kept
            arrays[f"v_{index:06d}"] = velocities
            arrays[f"k_{index:06d}"] = kept
            if (index + 1) % 100 == 0:
                print(f"{index + 1} / {len(names)}", flush=True)
    np.savez_compressed(output, **arrays)
    print(json.dumps({"images": len(names), "output": str(output)}))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
