from pathlib import Path

import numpy as np
import pytest

from dispick.physics.bank import BankConfig, ModalBank, compute_entry
from dispick.physics.dispersion import NormalizedModel, check_modes, model_curves, phase_velocities
from dispick.physics.earth import FAMILIES, EarthPrior, Family, LayeredModel, sample_model
from dispick.physics.excitation import group_velocity, vertical_excitation


def _two_layers(
    vs_top: float = 200.0, vs_bottom: float = 600.0, depth: float = 5.0
) -> LayeredModel:
    return LayeredModel(
        thickness=np.array([depth, 1.0]),
        vs=np.array([vs_top, vs_bottom]),
        vp=np.array([vs_top, vs_bottom]) * 2.0,
        rho=np.array([1800.0, 2000.0]),
        family=Family.NORMAL,
    )


def test_model_validation() -> None:
    with pytest.raises(ValueError, match="same length"):
        LayeredModel(np.ones(2), np.ones(3), np.ones(2) * 2, np.ones(2), Family.NORMAL)
    with pytest.raises(ValueError, match=r"1\.45"):
        LayeredModel(np.ones(2), np.ones(2), np.ones(2), np.ones(2), Family.NORMAL)
    model = _two_layers()
    assert model.depth == 5.0
    assert model.n_layers == 2
    assert model.tops.tolist() == [0.0, 5.0]


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_samples_a_valid_model(family: Family) -> None:
    prior = EarthPrior(family_weights={family: 1.0})
    rng = np.random.default_rng(0)
    for _ in range(20):
        model = sample_model(prior, rng)
        assert model.family == family
        # The half-space is the stiffest layer: every mode is guided.
        assert model.vs[-1] >= model.vs[:-1].max()
        assert np.all(model.vp >= 1.45 * model.vs * (1 - 1e-6))
        assert np.all((model.rho >= 1400) & (model.rho <= 2700))


def test_fundamental_mode_limits() -> None:
    """M0 tends to the half-space's Rayleigh velocity at long wavelengths and to the top
    layer's at short ones (about 0.93 Vs for Vp = 2 Vs)."""
    model = _two_layers()
    normalized = NormalizedModel.of(model)
    f_hat = np.array([0.002, 0.005, 200.0, 500.0])
    c = phase_velocities(normalized, f_hat, n_modes=1, dc=0.0005)[0] * normalized.velocity
    assert c[0] == pytest.approx(0.933 * 600, rel=0.01)
    assert c[-1] == pytest.approx(0.933 * 200, rel=0.01)
    # Normal dispersion: faster at lower frequencies (to rounding, at the asymptotes).
    assert np.all(np.diff(c[::-1]) >= -1e-4 * c[1:])


def test_check_modes_accepts_steep_curves_and_rejects_skips() -> None:
    f_hat = np.geomspace(0.1, 10, 200)
    smooth = np.vstack([0.5 + 0.4 / (1 + (f_hat / 0.5) ** 4), 0.95 + 0.0 * f_hat])
    assert check_modes(smooth, f_hat) == (2, None)
    skipped = smooth.copy()
    skipped[0, 120:] *= 1.3
    reliable, reason = check_modes(skipped, f_hat)
    assert reliable == 0
    assert reason is not None
    assert "jumps" in reason
    crossing = smooth.copy()
    crossing[1] = 0.4
    assert check_modes(crossing, f_hat)[0] == 1


def test_model_curves_keeps_m0_and_orders_modes() -> None:
    config = BankConfig()
    f_hat = config.f_hat()
    result = model_curves(_two_layers(), f_hat, n_modes=4)
    assert result is not None
    _, c, _ = result
    assert np.isfinite(c[0]).all()
    for mode in range(1, 4):
        both = np.isfinite(c[mode]) & np.isfinite(c[mode - 1])
        assert np.all(c[mode][both] > c[mode - 1][both])


def test_group_velocity_of_a_flat_curve_is_the_phase_velocity() -> None:
    f_hat = np.geomspace(0.1, 10, 50)
    c = np.full((1, 50), 0.4)
    assert np.allclose(group_velocity(c, f_hat), 0.4)


def test_excitation_is_relative_and_m0_dominates_normal_dispersion() -> None:
    config = BankConfig()
    f_hat = config.f_hat()
    result = model_curves(_two_layers(), f_hat, n_modes=3)
    assert result is not None
    normalized, c, dc = result
    a = vertical_excitation(normalized, f_hat, c, n_samples=16, dc=dc)
    assert a.shape == c.shape
    assert np.nanmax(a) == pytest.approx(1.0)
    assert np.all(np.isnan(a[~np.isfinite(c)]))
    # Steady normal dispersion: the fundamental mode carries most of the energy.
    assert np.nanmedian(a[0]) > 0.5


def test_bank_entries_are_deterministic_and_round_trip(bank_path: Path, bank: ModalBank) -> None:
    config = bank.config
    entry, _ = compute_entry(config, 5)
    stored = bank[5]
    assert np.allclose(stored.c_hat, entry.c_hat, equal_nan=True)
    assert np.allclose(stored.a_hat, entry.a_hat, equal_nan=True, rtol=2e-3, atol=1e-3)
    assert stored.model.family == entry.model.family
    lazy = ModalBank(bank_path)
    assert np.allclose(lazy[5].c_hat, stored.c_hat, equal_nan=True)
    assert len(lazy) == config.n_models


def test_bank_curves_scale_with_velocity_and_length(bank: ModalBank) -> None:
    entry = bank[0]
    f = np.geomspace(1, 100, 40)
    c1, _ = entry.curves(f)
    # Twice the velocities and twice the thicknesses: the same curve, twice as fast.
    c2, _ = entry.curves(f, velocity_scale=2.0, length_scale=2.0)
    assert np.allclose(c2[0], 2 * c1[0], rtol=1e-6)
    # Twice the thicknesses alone: the curve shifts to half the frequencies.
    c3, _ = entry.curves(f / 2, length_scale=2.0)
    assert np.allclose(c3[0], c1[0], rtol=1e-6)
