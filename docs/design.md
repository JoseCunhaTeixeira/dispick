# How dispick works

dispick picks the fundamental Rayleigh mode (M0) on phase-shift dispersion images and rates
whether an image can be picked at all. It learns from synthetic images only, made with the
same transform sigpipe applies to real records, so that it meets real images as one more
variation of what it trained on. This document explains each part and why it is built so.

## 1. What is learned

For an image `fv_map[n_f, n_v]` over its frequencies and velocities, the network gives:

- per frequency, a **distribution over velocity**: where M0 lies;
- per frequency, a **presence** probability: whether M0 can be picked there (see 4);
- per image, three numbers: **pickable** (a probability), **quality** (0 to 1) and
  **higher-mode share** (0 to 1), for gating windows before trusting a pick (see 4).

A distribution per frequency, rather than one regressed velocity, keeps the network honest
where two ridges compete: it can say "here or there", and the decoding (see 6) settles it with
the neighbouring frequencies. It also gives an uncertainty for free: the spread of the
distribution.

## 2. The synthetic images

### 2.1 The modal bank (`dispick.physics`)

Random 1D layered models of the near surface, in six families that give the network different
pictures (`physics/earth.py`):

| Family | Ground | What the image shows |
|---|---|---|
| `normal` | velocity increasing by steps | steady dispersion, M0 dominant |
| `gradient` | power-law increase, over bedrock or not | smooth curves, strong bedrock resonances |
| `stiff_top` | a stiff crust over softer ground | inverse dispersion at high frequency, higher modes taking the energy |
| `lvl` | a low-velocity layer at depth | osculating modes, abrupt M0 transitions |
| `bedrock` | soft cover over bedrock 2.5 to 10 times stiffer | staircase higher modes, mode kissing |
| `random` | a random walk in log(Vs) | everything else |

Surface Vs 60 to 700 m/s, depth to the half-space 1 to 60 m, Vp/Vs 1.6 to 2.8, an optional
water table (Vp of at least 1450 m/s below it), density from Vs. The half-space is always the
stiffest layer, so every mode is a guided mode.

For each model, disba computes six Rayleigh modes on 384 normalized frequencies
(`physics/dispersion.py`). Two things needed care:

- **Scale.** disba searches roots in steps of `dc` km/s; a model at 80 m/s would be searched
  in steps of 6 % of its velocity and close modes skipped. Models are normalized first (depth
  to the half-space 1 km, half-space Vs 1 km/s) and the step taken relative to the slowest
  layer. Dispersion is scale-invariant, so the curves are exact once scaled back.
- **Mode tracing.** disba (CPS surf96) traces each mode from the shortest period up, and at
  very short wavelengths higher modes crowd near the top layer's Vs, where the first root
  count can skip one. Curves are checked: modes ordered at every frequency, no isolated jump
  (a steep but smooth transition is physical, a lone step is a skip). When a higher mode fails,
  the modes are searched again below that crowded band, and held flat above it. A model whose
  M0 fails is dropped. On the default prior, about 80 % of the models keep all six modes and
  94 % at least one higher mode.

Each mode's **excitation** comes from the medium response of a vertical force on vertical
geophones (Aki & Richards, eq. 7.150): `uz(0)^2 / (c U I)`, with `I` the energy integral of
the eigenfunctions (`physics/excitation.py`). It is what makes M0 dominate on steadily
dispersive ground and hand its energy to higher modes above a stiff crust.

The bank stores the curves normalized. One entry then serves any velocity and length scale:
`c(f) = s_v V c_hat(f s_l H / (s_v V))`. 50 000 entries take about 40 minutes on 10 cores.

### 2.2 One image (`dispick.synthesis`)

Each training image draws, independently:

- **An array** (`acquisition.py`): 3 to 144 receivers (most 7 to 48), 0.1 to 5 m apart, a few
  per cent irregular; an active shot 0.03 to 3 apertures away, or a passive virtual shot on
  the first receiver (its trace vanishes in the transform, as in sigpipe); position errors.
- **A scale** fitting the model to the array: its depth to the half-space is 0.08 to 4 times
  the array's central wavelength `2 sqrt(spacing x aperture)`. Some images see only the top
  layer, some only the half-space, most the transition.
- **Axes as users set them** (`generator.py`): fmax where M0's wavelength is 0.5 to 12
  spacings (deep into aliasing, or stopping long before, like PAC's 100 Hz on 0.25 m
  spacings), often rounded; fmin 0 half of the time; sigpipe's frequency grid (multiples of
  df) with 12 to 600 frequencies, and 8 % of coarse grids of 3 to 12 (short passive
  segments); vmax 0.8 to 4 times M0's velocity at the longest resolved wavelength (the curve
  may leave the image), vmin at 0 or 1 m/s (PAC's) or below M0; 1000 velocities often
  (sigpipe's default), else 100 to 1500.
- **The records** (`wavefield.py`), as spectra at the image's frequencies:
  - active: each mode as the Hankel function `H0(2)(k r)`, which keeps the near field (M0 reads
    slow at wavelengths longer than the offsets);
  - passive: plane waves from 3 to 24 azimuths spread by up to 45 degrees (each reads
    `c / cos(theta)`: passive ridges smear upward), part from the far side, or a diffuse field;
  - mode amplitudes from the excitation (80 %) or random (20 %), each scattered by a factor
    and drifting along frequency; attenuation (Q 5 to 100);
  - ground varying along the array (15 %): a slowness contrast past a split point;
  - coherent noise: the air wave (330 to 345 m/s, active), body waves at the layers' P and S
    velocities, mains hum at 50 or 60 Hz and harmonics in phase on all traces (it peaks at the
    grid's top velocity, as on PAC's images), surface waves reflected back;
  - random noise at a signal-to-noise ratio shaped like a source band (peak -8 to 35 dB,
    roll-offs, ripple);
  - trace defects: coupling, dead traces, reversed polarities, timing errors;
  - stacking (30 %): 2 to 4 records averaged, noise redrawn (passive: sources too);
  - at 0 Hz, real values like sigpipe's DC bin (after normalization, only their signs are
    left: the 0 Hz column is flat, from 0 for demeaned records to 1 for a shared offset).
- **The transform** (`transform.py`): sigpipe's phase shift, step for step (sqrt(offset)
  weights, unit spectra, steering, division by the distinct offsets). Checked against sigpipe:
  the same image to 3e-6.

Some images have nothing to pick, on purpose: noise only (4 %), coherent noise only (2 %),
axes missing M0 (3 %); many more are unpickable by nature (a 5-receiver array at long
wavelengths, low SNR). About half the images are unpickable: only 10 % of those from 3 to 6
receivers are pickable (as PACo measured on the demo's 5-receiver windows), 28 % from 7 to 12,
46 % from 13 to 24, 67 % from 25 to 48 and 81 % from 49 up.

A training image takes about 35 ms of one core, so training draws fresh images endlessly
(`data/datasets.py`): the network never sees an image twice.

## 3. One network for any axes and any array

- **A fixed grid.** Each image is resampled onto 256 x 256 cells spanning its own axes
  (`grid.py`): a tent average when coarsening (so a 2000-velocity image is not aliased),
  symmetric at the edges; refining, velocities are interpolated linearly, while each
  frequency row copies the image's nearest column (a row between two columns would blend two
  ridges into one the image does not have). The training targets are made of the image's
  columns with the same weights, so a target always sits on the ridge its row shows. The picks
  come back to the image's own frequencies.
- **Where each cell stands physically** (`features.py`), as input channels: the wavelength over
  twice the spacing (below 1 the array aliases) and over the aperture (above 1 it hardly
  resolves velocity: a broad ridge is then normal, not noise). With the coherence over the
  noise floor 1/sqrt(N) and each frequency's column over its maximum, that is five channels.
  Everything else (units, ranges, steps) the network learns to ignore, having seen them all.
- **Unknown arrays.** 10 % of the training images hide their geometry (channels at 0): an
  image whose acquisition is unknown, like a stack of shots, is still picked.

## 4. What "pickable" means

The truth of a synthetic image is its modes' exact velocities, but a pick is only as good as
what the image shows. M0 is **pickable at a frequency** (`synthesis/labels.py`) when the image
has a ridge peak within 5 % of M0's velocity that stands out from chance, at a wavelength of
at least 2 spacings (the spatial Nyquist limit, as sigpipe's and PACo's pickers start):

- at least 10 % of the way from the noise floor to a perfect plane wave, and above what random
  phases reach 5 % of the time (a Rayleigh tail: sqrt(ln 20 / N) for N traces, counting only
  the traces the phase shift weighs, so not a virtual source's own);
- at least 30 % as high as the column's highest peak;
- over a continuous stretch: stretches shorter than 3 columns (or 1 % of the columns, on fine
  images) are dropped first, then single-column gaps between the stretches left are bridged.
  In that order: a random image has peaks everywhere, some near M0 by chance, and bridging
  first would chain them into stretches. On random records at 600 frequencies, under 1 % of
  the columns come out pickable.

The presence output learns that: it is the probability that a pick at that frequency lies
within 5 % of M0.

The image's labels:

- **pickable**: a continuous stretch of pickable M0 spans at least half an octave of
  wavelength (its longest wavelength 1.41 times its shortest);
- **quality**: that stretch's span over 3 octaves, capped at 1. What a curve is worth to an
  inversion is the depth range its wavelengths cover: 0.33 is one octave;
- **higher_mode_share**: the share of M0's frequencies where a higher mode is the column's
  highest peak: the image invites mode jumps.

## 5. The network (`model/`)

A U-Net over the 256 x 256 grid (widths 32 to 384, two residual blocks per stage, group
normalization), with two transformer layers over the coarsest (16 x 16) map: which ridge is M0
is a global question (the lowest coherent branch, continuous over the band, below its
siblings), which convolutions answer only locally. 16 M parameters, about 0.2 to 0.3 s an image
on a CPU.

Heads: the velocity logits (per frequency, softmax along velocity), presence (each
frequency's column of features pooled along velocity, then a 1D convolution along
frequency), and the image's three outputs (from the coarsest map).

Losses: cross-entropy of each column's velocity distribution against a Gaussian of 1.5 bins
around the true bin (weighted 1 where M0 is pickable, 0.2 where it is in the image but not
pickable: the network learns where the ridge would continue, and its presence says not to
trust it there), binary cross-entropy for presence and for the image labels.

Training: AdamW, warm-up and cosine decay, bf16 on GPUs, gradient clipping, and an exponential
moving average of the weights, which is what is validated and exported.

## 6. From outputs to a curve (`inference/`)

The velocity path is decoded jointly over frequency (Viterbi): each column's log-probability
weighed by its presence, a jump between neighbouring columns costing 0.1 per velocity bin. A
column hesitating between two ridges follows its neighbours. The chosen bin is refined to a
fraction of a bin, and its spread gives the uncertainty. The points kept are those whose
presence reaches the threshold (0.5), within the widest continuous stretch.

An optional second pass (`PickSettings(zoom=True)`) looks again at the picked band, on a grid
spanning only it: finer velocity bins when the curve was a thin line in a wide image.

The curve handed to sigpipe and PAC carries the Lorentzian uncertainties PAC's inversion
expects, from the array's resolving power; the network's own spread is also available.

## 7. How it is judged (`evaluation/`)

On a benchmark of synthetic images at their own axes, end to end:

- **precision**: of the points picked, the share within 5 % of M0;
- **recall**: of the pickable points, the share picked and within 5 %;
- **mode confusion**: picked points sitting on a higher mode instead of M0;
- **accuracy where pickable**: the velocity estimate alone, picked or not;
- the image verdict's accuracy, F1, ROC AUC and Brier score, and the quality's error;

overall and by array size, record kind, scenario, ground family, signal-to-noise and frequency
count. The classical baseline is sigpipe's `pick_curves` (the brightest velocity per frequency,
above the aliasing limit, smoothed), which picks every frequency.

## 8. Why M0 only

The bank holds six modes per model, and the network's velocity head has one channel per mode:
`network.n_modes` above 1 trains higher modes too, with no new data. M0 alone is the default
for three reasons:

- the physics labels higher modes by counting ridges, but images do not: where M0 fades (a
  stiff crust at high frequency), the lowest visible ridge is M1, and whether a ridge is M1
  or M2 depends on ridges the image may not show. The network would learn a numbering it
  cannot see;
- the inversion PAC and PACo run does well on M0 alone, and a wrong mode number hurts it more
  than a missing mode;
- M0 is where the classical picking was inconsistent.

What matters for M0 is that the network sees the higher modes, bright or faint, in every
image: it learns not to pick them. Multi-mode picking is a natural next step: train with
`n_modes: 3`, and judge it with the mode-confusion metric per mode.

## 9. Limits

- Everything learned is synthetic. The images are made as sigpipe makes them, with the
  defects of real records, but real ground has more (3D structure, topography, cultural
  noise). The first check on real data is the PACo demo profiles; the next step is a set of
  real images picked by hand to measure against, and to fine-tune on.
- Rayleigh waves on vertical geophones, phase velocity: not Love waves, not the group-velocity
  (FTAN) images sigpipe also makes.
- The quality rating is about M0's wavelength span: it does not see whether the processing
  was right (a wrong trigger shows as a shifted ridge, not as a bad image).
