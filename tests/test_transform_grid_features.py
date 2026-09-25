import numpy as np
import pytest

from dispick.features import CHANNELS, Geometry, input_channels, normalize_coherence
from dispick.grid import CanonicalGrid, from_canonical, resampling_matrix, to_canonical
from dispick.synthesis import transform
from dispick.synthesis.transform import phase_shift


def _plane_wave(offsets: np.ndarray, frequencies: np.ndarray, velocity: float) -> np.ndarray:
    return np.exp(-2j * np.pi * frequencies[None, :] * offsets[:, None] / velocity)


def test_phase_shift_peaks_at_the_wave_velocity() -> None:
    offsets = 1.0 + np.arange(24) * 0.5
    frequencies = np.linspace(5, 60, 12)
    velocities = np.linspace(50, 600, 551)
    image = phase_shift(_plane_wave(offsets, frequencies, 250.0), frequencies, offsets, velocities)
    assert image.shape == (12, 551)
    assert np.allclose(velocities[np.argmax(image, axis=1)], 250.0, atol=1.0)
    assert image.max() == pytest.approx(1.0, abs=1e-5)


def test_phase_shift_regular_and_direct_paths_agree() -> None:
    rng = np.random.default_rng(0)
    offsets = np.arange(10) * 0.7
    frequencies = np.linspace(0, 80, 17)
    velocities = np.linspace(8, 800, 97)  # no v = 0: its steering phases are rounding noise
    spectra = rng.standard_normal((10, 17)) + 1j * rng.standard_normal((10, 17))
    regular = phase_shift(spectra, frequencies, offsets, velocities)
    unit = spectra * np.sqrt(offsets)[:, None]
    unit = unit / (np.abs(unit) + 1e-12)
    direct = transform._direct(unit, frequencies, offsets, velocities) / offsets.size  # pyright: ignore[reportPrivateUsage]
    assert np.allclose(regular, direct, atol=1e-5)
    # sigpipe weighs traces by sqrt(offset): the zero-offset trace vanishes, still counted.
    assert regular.max() <= 9 / 10 + 1e-5


def test_phase_shift_matches_sigpipe() -> None:
    sigpipe = pytest.importorskip("sigpipe.algorithms.dispersion.phase_shift")
    from scipy.fft import rfft, rfftfreq

    rng = np.random.default_rng(1)
    offsets = (0.75 + 0.25 * np.arange(16)).astype(np.float32)
    xt = rng.standard_normal((16, 512))
    _, vs, reference = sigpipe.phase_shift(
        xt, 500.0, offsets, fmin=1, fmax=100, vmin=10, vmax=900, nv=120
    )
    frequencies = rfftfreq(512, 1 / 500.0)
    band = (frequencies >= 1) & (frequencies <= 100)
    spectra = np.asarray(rfft(xt, axis=1))
    ours = phase_shift(spectra[:, band], frequencies[band], offsets, vs.astype(float))
    assert np.max(np.abs(ours - reference)) < 1e-4


def test_resampling_interpolates_linearly_and_averages_when_coarsening() -> None:
    source = np.linspace(0, 10, 11)
    fine = np.linspace(0, 10, 101)
    weights = resampling_matrix(source, fine)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.allclose(weights @ (3 * source + 1), 3 * fine + 1)
    coarse = resampling_matrix(fine, source)
    alternating = np.where(np.arange(101) % 2 == 0, 1.0, -1.0)
    # Anti-aliased inside; the two ends are plain samples (a symmetric tent has no room there).
    assert np.all(np.abs((coarse @ alternating)[1:-1]) < 0.25)


def test_to_canonical_and_back() -> None:
    frequencies = np.linspace(0, 100, 201)
    velocities = np.linspace(1, 1000, 1000)
    image = np.add.outer(frequencies / 100, velocities / 1000) / 2
    canonical, axes = to_canonical(image, frequencies, velocities, CanonicalGrid(64, 128))
    assert canonical.shape == (64, 128)
    assert axes.f_range == (0.0, 100.0)
    assert axes.v_range == (1.0, 1000.0)
    expected = np.add.outer(axes.frequencies / 100, axes.velocities / 1000) / 2
    assert np.allclose(canonical, expected, atol=2e-3)
    bins = axes.bin_of_velocity(np.array([1.0, 500.5, 1000.0]))
    assert np.allclose(bins, [0.0, 63.5, 127.0])
    assert np.allclose(axes.velocity_of_bin(bins), [1.0, 500.5, 1000.0])
    values = np.linspace(0, 1, 64)
    values[10] = np.nan
    back = from_canonical(values, axes.frequencies, np.array([0.0, 50.0, axes.frequencies[10]]))
    assert np.isfinite(back[:2]).all()
    assert np.isnan(back[2])


def test_to_canonical_rejects_bad_axes() -> None:
    with pytest.raises(ValueError, match="increasing"):
        to_canonical(np.zeros((3, 3)), np.array([0, 2, 1]), np.arange(3), CanonicalGrid(8, 8))
    with pytest.raises(ValueError, match="does not match"):
        to_canonical(np.zeros((3, 4)), np.arange(3), np.arange(3), CanonicalGrid(8, 8))


def test_input_channels() -> None:
    grid = CanonicalGrid(16, 32)
    axes = grid.axes((0.0, 100.0), (1.0, 1000.0))
    image = np.random.default_rng(0).uniform(0, 1, (16, 32)).astype(np.float32)
    geometry = Geometry(n_receivers=24, spacing=0.5)
    channels = input_channels(image, axes, geometry)
    assert channels.shape == (len(CHANNELS), 16, 32)
    assert channels.dtype == np.float32
    assert np.all((channels[:2] >= 0) & (channels[:2] <= 1))
    assert np.allclose(channels[1].max(axis=1), 1.0)
    assert np.all(np.abs(channels[2:4]) <= 1.0)
    assert np.all(channels[4] == 1.0)
    # Wavelength = 2 x spacing is the aliasing limit: the alias channel crosses 0 there.
    f, v = axes.frequencies[8], axes.velocities
    first_positive = int(np.argmax(channels[2][8] > 0))
    assert v[first_positive - 1] / f <= 2 * geometry.spacing <= v[first_positive] / f
    unknown = input_channels(image, axes, None)
    assert np.all(unknown[2:] == 0.0)


def test_normalize_coherence_floor_and_foreign_ranges() -> None:
    image = np.array([[0.2, 0.6, 1.0]], dtype=np.float32)
    assert np.allclose(normalize_coherence(image, 0.2), [[0.0, 0.5, 1.0]])
    foreign = np.array([[0.0, 5.0, 10.0]], dtype=np.float32)
    assert np.allclose(normalize_coherence(foreign, 0.2), [[0.0, 0.5, 1.0]])


def test_geometry() -> None:
    geometry = Geometry.from_positions(np.array([0.0, 0.5, 1.0, 1.5]))
    assert geometry.n_receivers == 4
    assert geometry.spacing == pytest.approx(0.5)
    assert geometry.aperture == pytest.approx(1.5)
    assert geometry.noise_floor == pytest.approx(0.5)
    with pytest.raises(ValueError, match="at least 2"):
        Geometry(n_receivers=1, spacing=1.0)


def test_geometry_from_receivers_survives_folded_offsets() -> None:
    from dispick.features import geometry_from_receivers

    x = np.arange(24) * 1.0
    geometry = geometry_from_receivers(x)
    assert geometry is not None
    assert geometry.spacing == pytest.approx(1.0)
    assert geometry_from_receivers(np.zeros(5)) is None
    assert geometry_from_receivers(np.array([0.0, np.nan])) is None
    sigpipe = pytest.importorskip("sigpipe.base")
    from dispick.integrations.sigpipe import geometry_of

    # A source inside the spread folds sigpipe's offsets onto each other (review finding).
    for source in (11.5, 11.3):
        acquisition = sigpipe.LinearAcquisition(
            source=sigpipe.Coordinate(source, 0.0, 0.0),
            receivers=tuple(sigpipe.Coordinate(float(v), 0.0, 0.0) for v in x),
        )
        folded = geometry_of(acquisition)
        assert folded is not None
        assert folded.spacing == pytest.approx(1.0)
