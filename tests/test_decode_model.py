import numpy as np
import pytest
import torch

from dispick.inference.decode import decode, longest_run, refine, spread, viterbi
from dispick.model.losses import LossConfig, picking_loss, soft_targets
from dispick.model.network import DispersionNet, NetworkConfig
from dispick.training.metrics import Metrics, refined_bins


def test_viterbi_follows_a_ridge_through_an_ambiguous_column() -> None:
    n_f, n_v = 20, 50
    log_probs = np.full((n_f, n_v), -10.0)
    log_probs[:, 20] = 0.0
    log_probs[10, 20] = -3.0
    log_probs[10, 40] = 0.0  # one column prefers another ridge
    path = viterbi(log_probs, np.ones(n_f), jump_cost=0.5)
    assert np.all(path == 20)
    greedy = viterbi(log_probs, np.ones(n_f), jump_cost=0.0)
    assert greedy[10] == 40


def test_viterbi_ignores_columns_without_presence() -> None:
    log_probs = np.full((5, 30), -10.0)
    log_probs[:, 5] = 0.0
    log_probs[2, :] = -10.0
    log_probs[2, 25] = 0.0
    weights = np.array([1.0, 1.0, 0.0, 1.0, 1.0])
    assert viterbi(log_probs, weights, jump_cost=0.1)[2] == 5


def test_refine_and_spread() -> None:
    probabilities = np.zeros((1, 20))
    probabilities[0, 10] = 0.5
    probabilities[0, 11] = 0.5
    bins = refine(probabilities, np.array([10]))
    assert bins[0] == pytest.approx(10.5)
    assert spread(probabilities, bins, window=5)[0] == pytest.approx(0.5)


def test_decode_returns_fractional_bins_and_presence() -> None:
    logits = np.full((8, 32), -20.0)
    logits[:, 12] = 5.0
    logits[:, 13] = 5.0
    bins, sigma, presence = decode(logits, np.array([10.0] * 4 + [-10.0] * 4))
    assert np.allclose(bins, 12.5, atol=1e-3)
    assert np.all(sigma < 1.0)
    assert np.all(presence[:4] > 0.99)
    assert np.all(presence[4:] < 0.01)
    with pytest.raises(ValueError, match="unknown decoding"):
        decode(logits, np.zeros(8), method="magic")


def test_longest_run_bridges_small_gaps() -> None:
    frequencies = np.arange(12.0)
    mask = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0], dtype=bool)
    kept = longest_run(mask, frequencies, max_gap=1)
    assert kept.tolist() == [True, True, False, True, True] + [False] * 7
    assert not longest_run(np.zeros(5, dtype=bool), np.arange(5.0)).any()


def _tiny() -> DispersionNet:
    return DispersionNet(
        NetworkConfig(widths=(8, 16, 32), attention_layers=1, attention_heads=2, groups=4)
    )


def test_network_shapes_and_grid_check() -> None:
    network = _tiny()
    outputs = network(torch.zeros(2, 5, 32, 48))
    assert outputs["logits"].shape == (2, 1, 32, 48)
    assert outputs["presence"].shape == (2, 1, 32)
    assert outputs["image"].shape == (2, 3)
    with pytest.raises(ValueError, match="divisible"):
        network(torch.zeros(1, 5, 30, 48))


def _batch(n_f: int = 32, n_v: int = 48) -> dict[str, torch.Tensor]:
    bins = torch.full((2, 2, n_f), float("nan"))
    bins[:, 0, 4:28] = 20.0
    bins[:, 1, 4:28] = 35.0
    presence = torch.zeros(2, n_f)
    presence[:, 6:26] = 1.0
    return {
        "inputs": torch.rand(2, 5, n_f, n_v),
        "target_bins": bins,
        "presence": presence,
        "image_targets": torch.tensor([[1.0, 0.6, 0.1], [0.0, 0.0, 0.0]]),
        "v_range": torch.tensor([[1.0, 1000.0], [10.0, 500.0]], dtype=torch.float64),
    }


def test_loss_is_finite_and_learns() -> None:
    torch.manual_seed(0)
    network = _tiny()
    batch = _batch()
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-3)
    losses: list[float] = []
    for _ in range(30):
        loss, parts = picking_loss(network(batch["inputs"]), batch, LossConfig())
        assert torch.isfinite(loss)
        losses.append(parts["loss"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert losses[-1] < 0.7 * losses[0]


def test_soft_targets_and_metrics() -> None:
    targets = soft_targets(torch.tensor([[3.0, float("nan")]]), 10, 1.0)
    assert torch.allclose(targets[0, 0].sum(), torch.tensor(1.0))
    assert targets[0, 0].argmax() == 3
    assert torch.all(targets[0, 1] == 0)
    probabilities = torch.zeros(1, 10)
    probabilities[0, 4] = probabilities[0, 5] = 0.5
    assert refined_bins(probabilities)[0] == pytest.approx(4.5)

    batch = _batch()
    logits = torch.full((2, 1, 32, 48), -20.0)
    logits[:, 0, :, 20] = 10.0
    outputs = {
        "logits": logits,
        "presence": torch.where(batch["presence"] > 0, 10.0, -10.0)[:, None],
        "image": torch.tensor([[5.0, 0.0, -2.0], [-5.0, -5.0, -5.0]]),
    }
    metrics = Metrics()
    metrics.update(outputs, batch)
    summary = metrics.summary()
    assert summary["acc_5"] == pytest.approx(1.0)
    assert summary["pick_precision"] == pytest.approx(1.0)
    assert summary["recall"] == pytest.approx(1.0)
    assert summary["mode_confusion"] == pytest.approx(0.0)
    assert summary["pickable_accuracy"] == pytest.approx(1.0)
    # Picking the higher mode instead is a mode confusion.
    logits[:, 0, :, :] = -20.0
    logits[:, 0, :, 35] = 10.0
    confused = Metrics()
    confused.update(outputs | {"logits": logits}, batch)
    assert confused.summary()["mode_confusion"] == pytest.approx(1.0)
