# Using dispick in sigpipe, PAC and PACo

dispick's adapter takes and returns sigpipe's own types, so each tool plugs it in with a few
lines. Nothing below is applied yet in those repositories: this is the plan, with the code.

The adapter (`dispick.integrations.sigpipe`):

```python
from dispick.integrations.sigpipe import assess_dispersion_image, pick_dispersion_image, pick_image

image = pick_dispersion_image(image)  # a DispersionImage, with M0 added to its curves
assessment = assess_dispersion_image(image)  # verdict, pickable, quality, higher_mode_share
result = pick_image(image)  # everything: per-frequency presence, spread, ...
```

- The curve is labelled like PAC's, `Mode("M", 0)`, with the image's velocity type and
  acquisition, and the Lorentzian uncertainties PAC's inversion expects (`uncertainty="model"`
  for the network's own spread, `"max"` for the larger of the two).
- `resample_over_wavelength=True` resamples it every metre of wavelength, as PAC's box picks.
- An image the network rates unpickable, or with fewer than 2 points, comes back unchanged
  (`on_unpickable="skip"`), raises (`"raise"`), or gets its points anyway (`"keep"`).
- The array (receiver count and spacing) comes from the image's acquisition. An unknown
  acquisition (a stack of shots) is picked without it: the network was trained for that too.
- The model is found in `$DISPICK_MODEL` or in dispick's package (`dispick/models/`); inference
  needs numpy and onnxruntime only (`dispick[onnx]`), no PyTorch.

## sigpipe: a `Pick` method

In `src/sigpipe/algorithms/picking/registry.py`, a method imported on use, like silex:

```python
def _pick_dispick(dispersion_image: DispersionImage, **params: object) -> DispersionImage:
    # Deferred import: dispick is an optional dependency (sigpipe[dispick]).
    from dispick.integrations.sigpipe import pick_dispersion_image

    return pick_dispersion_image(dispersion_image, **params)  # type: ignore[arg-type]


DISPERSION_PICKING_METHODS: dict[str, Callable[..., DispersionImage]] = {
    "maximum": pick_curves,
    "dispick": _pick_dispick,
}
```

In `transformers/picking.py`, `method: Literal["none", "maximum", "dispick"]`; in
`pyproject.toml`:

```toml
[project.optional-dependencies]
dispick = ["dispick[onnx] @ git+https://github.com/JoseCunhaTeixeira/dispick"]
```

Then, in a pipeline:

```python
>> Dispersion(method="phase", fmin=0, fmax=100, vmin=1, vmax=1000)
>> Pick(method="dispick", threshold=0.5, resample_over_wavelength=True)
```

## PAC: an automatic pick the user corrects

PAC's picking is interactive (`masw/algorithms/dispersion_picking.py`, the lasso). An
"Auto-pick" action per window, or for the whole line:

```python
from dispick.integrations.sigpipe import pick_image, pick_dispersion_image


def pick_curve_auto(dispersion_image: DispersionImage) -> tuple[DispersionImage, dict]:
    result = pick_image(dispersion_image)
    picked = pick_dispersion_image(dispersion_image, resample_over_wavelength=True)
    return picked, result.image.to_dict()  # the verdict, for the UI
```

The UI shows the verdict next to each window (pickable, doubtful, unpickable) and the curve as
an ordinary pick, which the lasso can replace. Command line, on PAC's output folders directly:

```sh
dispick pick data/output/<profile>/xmid_*/DispersionImage_0000.hdf5 --out picks/ --figure
```

writes, per window, `xmid_<x>_DispersionImage_0000_M0.csv` in sigpipe's format (PAC reads it
like its own) and a JSON of the verdict.

## PACo: the picker behind `pick`, and the verdict in the gates

PACo's agent cannot see images: it reads the gates' summaries. dispick gives the gates two
things the classical tracker could not: a calibrated verdict on each image, and a presence
probability per point.

**The picker.** `paco.picking.pick_modes` returns `PickedMode`s; a dispick version fills the
same fields, so G3 and G4 judge it unchanged:

```python
import numpy as np
from dispick.integrations.sigpipe import pick_image

from paco.picking.models import PickedMode, PickingParameters
from paco.picking.modes import _curve


def pick_modes_dispick(image: DispersionImage, parameters: PickingParameters) -> list[PickedMode]:
    result = pick_image(image)
    estimated = np.isfinite(result.velocities) & (result.frequencies > 0)
    if result.n_picked < parameters.min_frequencies:
        return []
    fs, vs = result.frequencies[estimated], result.velocities[estimated]
    rows = np.searchsorted(image.fs, fs)
    columns = np.clip(np.searchsorted(image.vs, vs), 0, image.vs.size - 1)
    kept = result.picked[estimated]
    return [
        PickedMode(
            number=0,
            frequencies=fs,
            velocities=vs,
            coherence=image.fv_map[rows, columns],
            pinned=np.zeros(fs.size, dtype=bool),
            kept=kept,
            noise_floor=1 / np.sqrt(len(image.acquisition.receivers)),
            curve=_curve(image, fs[kept], vs[kept], 0),
        )
    ]
```

with a `picker: "dispick" | "tracker"` choice in `PickingParameters` (the tracker kept as the
fallback and for comparison), and `max_modes > 1` staying with the tracker.

**G2, the image gate.** A new metric, `pickable`, the network's probability: below 0.3 the
window is rejected (`unpickable`, not fixable by picking again: a longer window or other
processing is the action), 0.3 to 0.7 flagged `doubtful`, kept. `higher_mode_share` above 0.3
flags `higher_modes_dominate`, the risk of a mode jump.

**G3, the curve gate.** The per-point presence and spread come with the pick: the median
presence of the kept points, and the share of points whose spread exceeds 10 % of their
velocity, are the curve's confidence; `max_jump` and `constant_wavelength` stay as they are.

**S2, the window length.** The coherence rules pick the window length by the ladder's G3 pass
counts. dispick's `quality` measures directly what a window length buys, the wavelength span
M0 can be picked over: the ladder can keep the shortest length whose median quality over the
line is within 10 % of the best.

## Versions

A model file carries its card (`dispick.json`): the grid, the channels, the dispick version,
the label definitions, the synthesis configuration and the validation metrics it was trained
with. A tool can log the card's version with each pick, so a curve is traceable to the model
that made it.
