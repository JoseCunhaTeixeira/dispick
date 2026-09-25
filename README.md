# dispick

Automatic picking of surface-wave dispersion curves on phase-shift dispersion images, by a
neural network trained on physics-based synthetic data only.

For each image, whatever its frequency and velocity ranges, steps and receiver array, dispick
gives:

- **the fundamental Rayleigh mode (M0)** at each of the image's frequencies, with an
  uncertainty and the probability that it can be picked there; the curve is the points kept;
- **a verdict on the image**: whether it can be picked at all (`pickable`, `doubtful`,
  `unpickable`), a quality score (the wavelength span M0 can be picked over) and the share of
  frequencies where a higher mode dominates. [PACo](../PACo) can gate its windows on it.

It is built for [sigpipe](../sigpipe)'s images, and so for [PAC](../PAC) and PACo: it reads
their `DispersionImage`s and HDF5 files, writes their curve format, and runs without PyTorch
(an ONNX model and onnxruntime).

```
records --sigpipe--> phase-shift image --dispick--> M0 curve + verdict --> inversion
```

## Install

Python 3.14 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --extra onnx                  # picking with an exported model
uv sync --extra onnx --extra sigpipe  # and the sigpipe adapter
```

Making data and training need more: `--extra synth --extra train` and one PyTorch build,
`--extra cpu`, `--extra cuda` (NVIDIA) or `--extra rocm` (AMD).

## Pick

A model is a pair of files, `dispick.onnx` and its card `dispick.json`. dispick looks for it in
`$DISPICK_MODEL`, then in its package (`src/dispick/models/`).

```python
from dispick import Geometry, Picker

picker = Picker.load("models/dispick.onnx")
result = picker.pick(fv_map, fs, vs, Geometry(n_receivers=24, spacing=0.25))

result.image.verdict  # "pickable", "doubtful" or "unpickable"
result.image.quality  # 0 to 1: the picked wavelengths' span over 3 octaves
frequencies, velocities, uncertainties = result.curve
result.presence  # per frequency: probability M0 is pickable there
```

With sigpipe's types (see [docs/integration.md](docs/integration.md)):

```python
from dispick.integrations.sigpipe import pick_dispersion_image

image = pick_dispersion_image(image)  # M0 added to the image's curves, as PAC labels it
```

On sigpipe/PAC files, from the command line:

```sh
dispick pick data/output/<profile>/xmid_*/DispersionImage_0000.hdf5 --out picks/ --figure
```

writes each window's curve in sigpipe's CSV format (with PAC's Lorentzian uncertainties), a
JSON of the verdict, and a figure of all the picks.

## How it works

In short (details and reasons in [docs/design.md](docs/design.md)):

1. **A modal bank**: 50 000 random near-surface earth models in six families (steady
   dispersion, gradients, stiff crusts, low-velocity layers, bedrock, random), with six Rayleigh
   modes each from [disba](https://github.com/keurfonluu/disba), checked for skipped roots, and
   each mode's excitation by a vertical force (the medium response). Stored normalized: one
   model serves any scale.
2. **Synthetic images**, drawn fresh for every training step: a model scaled to a random
   array (3 to 144 receivers, active or passive), axes set as users set them, records
   synthesized mode by mode (Hankel functions with their near field for shots, plane waves from
   many azimuths for passive virtual shots) with air waves, body waves, mains hum, reflections,
   random noise, dead or reversed traces, timing errors, stacking; transformed by sigpipe's
   phase shift exactly (checked to 3e-6).
3. **Labels** from what the image shows, not only from the physics: M0 is pickable where a
   ridge peaks within 5 % of it, clearly above the noise, unaliased; an image is pickable
   where such a stretch spans half an octave of wavelength.
4. **One network for any axes and arrays**: every image is resampled onto 256 x 256 cells over
   its own axes, with channels saying where each cell stands physically (its wavelength over
   the spacing: aliasing; over the aperture: resolution). A U-Net with a transformer at its
   core outputs a velocity distribution and a presence per frequency, and the image's verdict.
5. **Decoding**: the velocity path is chosen jointly over frequency (Viterbi), refined within a
   bin, and brought back to the image's own frequencies.

## Train

The pipeline, step by step (paths are examples):

```sh
dispick bank build --config configs/bank.yaml --out data/bank.h5 --workers 11
dispick data shard --bank data/bank.h5 --out data/validation.h5 --count 4096 --seed 1001 --workers 11
dispick data benchmark --bank data/bank.h5 --out data/benchmark.h5 --count 2000 --seed 2002 --workers 11
dispick data preview --bank data/bank.h5 --out preview.png          # look at the images
dispick train --config configs/train.yaml                           # resumes from runs/base/last.pt
dispick export --checkpoint runs/base/best.pt --out models/dispick.onnx
dispick evaluate --model models/dispick.onnx --benchmark data/benchmark.h5 --out report/ --baselines
```

Training wants a GPU: see [docs/cloud.md](docs/cloud.md) for AWS (EC2, SageMaker) and GCP
(Compute Engine, Vertex AI), with the Docker image (`docker/Dockerfile`) that runs the whole
pipeline, resumable, one process per GPU. About 3 to 4 hours on one L4 or A10G.

`configs/train_smoke.yaml` checks the whole chain on a CPU in minutes.

Every setting is a validated configuration (pydantic), saved with the data and the model:
`dispick.synthesis.config.SynthesisConfig` for the images, `dispick.training.config.TrainConfig`
for training (overridable from the command line: `--set optim.steps=100000`).

## Evaluate

`dispick evaluate` scores a model end to end on a benchmark and writes `report.md`,
`report.json` and per-image scores: precision (picked points within 5 % of M0), recall
(pickable points found), mode confusion (picked points on a higher mode), the velocity accuracy
where M0 is pickable, and the image verdict's accuracy, F1 and ROC AUC; overall and by array
size, record kind, ground family, noise and frequency count, next to sigpipe's classical
maximum picker (`--baselines`).

## Project layout

```
src/dispick/
  physics/      earth models, Rayleigh modes (disba), excitation, the modal bank
  synthesis/    arrays, wavefields and noise, the phase shift, labels, the generator
  grid.py       the canonical grid and resampling      features.py  the input channels
  data/         training examples, online datasets, shards and benchmarks
  model/        the network, the losses
  training/     configuration, the training loop, validation metrics
  inference/    decoding, ONNX/PyTorch backends, the Picker
  evaluation/   benchmarks, baselines, figures
  integrations/ sigpipe (and so PAC and PACo)
  io.py         sigpipe/PAC files without sigpipe      export.py    ONNX export
configs/        bank and training configurations
docker/, scripts/  the cloud training image and its pipeline
docs/           design, cloud training, integration
```

## Develop

```sh
uv sync --extra synth --extra train --extra onnx --extra plot --extra sigpipe --extra cpu
uv run ruff check && uv run ruff format --check
uv run pyright src tests
uv run pytest
```
