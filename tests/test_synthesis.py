import collections

import numpy as np
import pytest

from dispick.data.examples import make_example
from dispick.features import Geometry
from dispick.grid import CanonicalGrid
from dispick.synthesis.config import LabelConfig, SynthesisConfig
from dispick.synthesis.generator import SyntheticGenerator, nice
from dispick.synthesis.labels import clean_runs, image_labels, m0_visibility, runs
from dispick.synthesis.transform import phase_shift
from dispick.synthesis.wavefield import Heterogeneity, hankel2


def test_samples_are_deterministic(generator: SyntheticGenerator) -> None:
    a = generator.sample(np.random.default_rng(7))
    b = generator.sample(np.random.default_rng(7))
    assert np.array_equal(a.fv_map, b.fv_map)
    assert np.array_equal(a.visible, b.visible)
    assert a.info == b.info


def test_samples_look_like_sigpipe_images(generator: SyntheticGenerator) -> None:
    rng = np.random.default_rng(3)
    for _ in range(30):
        sample = generator.sample(rng)
        n_f, n_v = sample.fv_map.shape
        assert (n_f, n_v) == (sample.frequencies.size, sample.velocities.size)
        assert n_f >= 3
        assert n_v >= 64
        assert sample.fv_map.dtype == np.float32
        assert np.isfinite(sample.fv_map).all()
        assert sample.fv_map.min() >= 0
        assert sample.fv_map.max() <= 1.0 + 1e-5
        # sigpipe's axes: multiples of df, and linspace(vmin, vmax, nv).
        steps = np.diff(sample.frequencies)
        assert np.allclose(steps, steps[0])
        assert np.allclose(np.diff(sample.velocities), np.diff(sample.velocities)[0])
        assert sample.curves.shape[1] == n_f
        assert not sample.visible[sample.frequencies <= 0].any()
        assert sample.labels.quality <= 1.0
        if not sample.labels.pickable:
            assert sample.labels.quality == 0.0


def test_scenarios_without_m0_are_unpickable(bank: object) -> None:
    from dispick.physics.bank import ModalBank
    from dispick.synthesis.config import ScenarioPrior

    assert isinstance(bank, ModalBank)
    config = SynthesisConfig(scenarios=ScenarioPrior(noise_only_probability=1.0))
    sample = SyntheticGenerator(bank, config).sample(np.random.default_rng(0))
    assert sample.info["scenario"] == "noise_only"
    assert not sample.labels.pickable
    assert not sample.visible.any()
    config = SynthesisConfig(scenarios=ScenarioPrior(off_range_probability=1.0))
    rng = np.random.default_rng(1)
    for _ in range(5):
        sample = SyntheticGenerator(bank, config).sample(rng)
        inside = (sample.curves[0] >= sample.velocities[0]) & (
            sample.curves[0] <= sample.velocities[-1]
        )
        assert not inside.any()
        assert not sample.labels.pickable


def test_the_scenario_mix_has_pickable_and_unpickable_images(generator: SyntheticGenerator) -> None:
    rng = np.random.default_rng(11)
    counts = collections.Counter(generator.sample(rng).labels.pickable for _ in range(60))
    assert counts[True] > 10
    assert counts[False] > 5


def test_visibility_on_a_clean_ridge() -> None:
    offsets = 2.0 + np.arange(48) * 0.5
    frequencies = np.linspace(0, 80, 81)
    velocities = np.linspace(1, 800, 800)
    c0 = 150 + 250 * np.exp(-frequencies / 15)
    spectra = np.exp(-2j * np.pi * frequencies[None, :] * offsets[:, None] / c0[None, :])
    spectra[:, 0] = 1.0
    image = phase_shift(spectra, frequencies, offsets, velocities)
    geometry = Geometry(48, 0.5)
    config = LabelConfig()
    visible = m0_visibility(image, frequencies, velocities, c0, geometry, config)
    # Pickable from the first frequencies the array resolves up to the aliasing limit.
    lam = c0 / np.maximum(frequencies, 1e-9)
    resolvable = (frequencies > 0) & (lam >= 2 * geometry.spacing)
    assert visible[resolvable].mean() > 0.9
    assert not visible[~resolvable].any()
    labels = image_labels(image, frequencies, velocities, c0[None, :], visible, geometry, config)
    assert labels.pickable
    assert labels.quality > 0.5
    assert labels.higher_mode_share == 0.0


def test_visibility_ignores_a_curve_off_the_ridge() -> None:
    offsets = np.arange(24) * 1.0 + 1
    frequencies = np.linspace(1, 50, 50)
    velocities = np.linspace(10, 1000, 500)
    spectra = np.exp(-2j * np.pi * frequencies[None, :] * offsets[:, None] / 300.0)
    image = phase_shift(spectra, frequencies, offsets, velocities)
    wrong = np.full(50, 500.0)
    visible = m0_visibility(image, frequencies, velocities, wrong, Geometry(24, 1.0), LabelConfig())
    assert not visible.any()


def test_runs_and_cleaning() -> None:
    mask = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    assert runs(mask) == [(1, 2), (4, 6), (10, 10)]
    cleaned = clean_runs(mask, LabelConfig(min_run=3, gap=0.05))
    # The one-column gap is closed, the lone column dropped.
    assert runs(cleaned) == [(1, 6)]


def test_make_example_targets(generator: SyntheticGenerator) -> None:
    rng = np.random.default_rng(5)
    grid = CanonicalGrid(64, 64)
    for _ in range(10):
        sample = generator.sample(rng)
        example = make_example(sample, grid, n_modes=3)
        assert example.image.shape == (64, 64)
        assert example.target_bins.shape == (3, 64)
        finite = np.isfinite(example.target_bins)
        assert np.all((example.target_bins[finite] >= 0) & (example.target_bins[finite] <= 63))
        assert np.all(finite[0][example.presence > 0])
        assert example.inputs().shape == (5, 64, 64)
        assert np.array_equal(example.image_targets, sample.labels.as_array())


def test_nice_numbers() -> None:
    assert nice(97.0) == 100.0
    assert nice(0.034) == 0.03
    assert nice(1234.0, "up") == 1500.0
    assert nice(1234.0, "down") == 1200.0
    assert nice(0.0) == 0.0


def test_hankel_far_field_and_heterogeneity() -> None:
    z = np.array([50.0, 200.0])
    far = np.sqrt(2 / (np.pi * z)) * np.exp(-1j * (z - np.pi / 4))
    assert np.allclose(hankel2(z), far, rtol=5e-3)
    offsets = np.linspace(0, 10, 11)
    assert Heterogeneity(split=100.0, contrast=1.2).factor(offsets) == pytest.approx(1.0)
    # All the array past the split: its velocity divided by the slowness ratio.
    assert Heterogeneity(split=-1.0, contrast=1.25).factor(offsets) == pytest.approx(0.8)


def test_heterogeneity_ridge_sits_on_the_effective_velocity() -> None:
    """The records follow the first part's velocity and the change past the split; the image's
    ridge, and so the label, is the factor times the first part's velocity."""
    from dispick.synthesis.config import WavefieldPrior
    from dispick.synthesis.wavefield import Modes, surface_waves

    offsets = 5.0 + np.arange(48) * 1.0
    frequencies = np.linspace(10, 40, 7)
    heterogeneity = Heterogeneity(split=25.0, contrast=1.25)
    first = np.full((1, 7), 300.0)
    modes = Modes(velocities=first, amplitudes=np.ones((1, 7)), q=np.array([1e9]))
    spectra = surface_waves(
        frequencies,
        offsets,
        modes,
        "active",
        WavefieldPrior(),
        np.random.default_rng(0),
        heterogeneity,
    )
    velocities = np.linspace(150, 450, 3001)
    image = phase_shift(spectra, frequencies, offsets, velocities)
    ridge = velocities[np.argmax(image, axis=1)]
    expected = 300.0 * heterogeneity.factor(offsets)
    assert np.allclose(ridge, expected, rtol=0.01)
