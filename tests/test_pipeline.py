"""The whole chain on a tiny scale: data, training, resume, export, picking, adapters."""

from pathlib import Path

import numpy as np
import pytest

from dispick.cli import main
from dispick.data.generate import DatasetSpec, build_benchmark, build_shard
from dispick.data.shards import Shard, read_benchmark
from dispick.evaluation.baselines import maximum_method
from dispick.evaluation.benchmark import evaluate, picker_method, write_report
from dispick.inference.picker import Picker, PickSettings
from dispick.synthesis.config import SynthesisConfig
from dispick.training.config import TrainConfig
from dispick.training.trainer import load_checkpoint, train

GRID = (32, 32)


@pytest.fixture(scope="module")
def datasets(bank_path: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    folder = tmp_path_factory.mktemp("data")
    spec = DatasetSpec(bank=bank_path, synthesis=SynthesisConfig(), seed=9, grid=GRID)
    shard = build_shard(spec, folder / "validation.h5", count=16, workers=2)
    benchmark = build_benchmark(spec, folder / "benchmark.h5", count=12, workers=1)
    return shard, benchmark


def _config(bank_path: Path, shard: Path, output: Path, steps: int) -> TrainConfig:
    return TrainConfig.model_validate(
        {
            "output": output,
            "data": {
                "bank": bank_path,
                "validation": shard,
                "grid": GRID,
                "batch_size": 4,
                "workers": 0,
            },
            "network": {
                "widths": [8, 16, 32],
                "attention_layers": 1,
                "attention_heads": 2,
                "groups": 4,
            },
            "optim": {"steps": steps, "warmup_steps": 2},
            "runtime": {
                "device": "cpu",
                "log_every": 2,
                "validate_every": 3,
                "checkpoint_every": 3,
            },
        }
    )


@pytest.fixture(scope="module")
def trained(
    bank_path: Path, datasets: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    output = tmp_path_factory.mktemp("run")
    shard, _ = datasets
    train(_config(bank_path, shard, output, steps=4))
    return output


def test_shards_and_benchmarks_round_trip(datasets: tuple[Path, Path]) -> None:
    shard_path, benchmark_path = datasets
    shard = Shard(shard_path)
    assert len(shard) == 16
    example = shard[3]
    assert example.image.shape == GRID
    assert float(example.image.min()) >= 0.0 and float(example.image.max()) <= 1.0
    samples = list(read_benchmark(benchmark_path))
    assert len(samples) == 12
    assert samples[0].fv_map.shape == (samples[0].frequencies.size, samples[0].velocities.size)


def test_training_writes_checkpoints_and_resumes(
    trained: Path, bank_path: Path, datasets: tuple[Path, Path]
) -> None:
    assert (trained / "last.pt").exists()
    assert (trained / "best.pt").exists()
    assert (trained / "config.yaml").exists()
    lines = (trained / "metrics.jsonl").read_text().splitlines()
    assert any('"split": "validation"' in line for line in lines)
    assert load_checkpoint(trained / "last.pt")["step"] == 4
    # Resuming carries on from step 4 to 6.
    train(_config(bank_path, datasets[0], trained, steps=6))
    assert load_checkpoint(trained / "last.pt")["step"] == 6
    config = TrainConfig.model_validate(load_checkpoint(trained / "last.pt")["config"])
    assert config.optim.steps == 6


def test_config_from_yaml_resolves_paths_and_overrides(tmp_path: Path) -> None:
    (tmp_path / "train.yaml").write_text("output: runs/a\ndata:\n  bank: data/bank.h5\n")
    config = TrainConfig.from_yaml(tmp_path / "train.yaml", {"optim.steps": 12})
    assert config.output == (tmp_path / "runs/a").resolve()
    assert config.data.bank == (tmp_path / "data/bank.h5").resolve()
    assert config.optim.steps == 12
    with pytest.raises(ValueError, match="divisible"):
        TrainConfig.model_validate({"output": "x", "data": {"bank": "b", "grid": [100, 100]}})


def test_export_pick_evaluate(trained: Path, datasets: tuple[Path, Path], tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    from dispick.export import export_onnx

    model = export_onnx(trained / "best.pt", tmp_path / "model.onnx")
    assert model.with_suffix(".json").exists()
    samples = list(read_benchmark(datasets[1]))
    for path in (model, trained / "best.pt"):
        picker = Picker.load(path)
        sample = samples[0]
        result = picker.pick(sample.fv_map, sample.frequencies, sample.velocities, sample.geometry)
        assert result.frequencies.shape == sample.frequencies.shape
        assert result.image.verdict in ("pickable", "doubtful", "unpickable")
        assert 0.0 <= result.image.pickable <= 1.0
        assert np.all(result.presence[result.picked] >= result.threshold)
        zoomed = picker.pick(
            sample.fv_map, sample.frequencies, sample.velocities, None, PickSettings(zoom=True)
        )
        assert zoomed.velocities.shape == sample.frequencies.shape
    picker = Picker.load(model)
    scores = evaluate(
        samples, {"dispick": picker_method(picker), "maximum": maximum_method}, chunk=5
    )
    reports = write_report(scores, tmp_path / "report")
    assert set(reports) == {"dispick", "maximum"}
    assert (tmp_path / "report" / "report.md").read_text().startswith("# dispick benchmark")


def test_sigpipe_adapter_and_files(
    trained: Path, datasets: tuple[Path, Path], tmp_path: Path
) -> None:
    pytest.importorskip("sigpipe")
    from sigpipe.base import Coordinate, DispersionImage, LinearAcquisition, Mode, VelocityType
    from sigpipe.dataio.dispersion.loading import load_dispersion_curves
    from sigpipe.dataio.dispersion.saving import save_dispersion_image

    from dispick.integrations.sigpipe import geometry_of, pick_dispersion_image
    from dispick.io import pick_files, read_image

    sample = next(iter(read_benchmark(datasets[1])))
    acquisition = LinearAcquisition(
        source=Coordinate(-1.0, 0.0, 0.0),
        receivers=tuple(
            Coordinate(float(x), 0.0, 0.0)
            for x in np.arange(sample.geometry.n_receivers) * sample.geometry.spacing
        ),
    )
    image = DispersionImage(
        fv_map=sample.fv_map,
        fs=sample.frequencies,
        vs=sample.velocities,
        type=VelocityType.PHASE,
        acquisition=acquisition,
    )
    geometry = geometry_of(acquisition)
    assert geometry is not None
    assert geometry.n_receivers == sample.geometry.n_receivers
    assert geometry.spacing == pytest.approx(sample.geometry.spacing, rel=1e-5)

    picker = Picker.load(trained / "best.pt")
    picked = pick_dispersion_image(image, picker=picker, threshold=0.0, on_unpickable="keep")
    assert picked.dispersion_curves is not None
    curve = picked.dispersion_curves[0]
    assert curve.mode == Mode("M", 0)
    assert curve.vs_err is not None
    assert curve.fs.size >= 2
    untouched = pick_dispersion_image(image, picker=picker, threshold=1.1)
    assert untouched.dispersion_curves is None
    with pytest.raises(ValueError, match="not pickable"):
        pick_dispersion_image(image, picker=picker, threshold=1.1, on_unpickable="raise")

    folder = tmp_path / "xmid_1.00"
    folder.mkdir()
    save_dispersion_image(image, folder / "DispersionImage_0000")
    stored = read_image(folder / "DispersionImage_0000.hdf5")
    assert stored.geometry is not None
    lines = list(
        pick_files(
            [folder / "DispersionImage_0000.hdf5"], tmp_path / "out", trained / "best.pt", 0.0
        )
    )
    assert len(lines) == 1
    csv = tmp_path / "out" / "xmid_1.00_DispersionImage_0000_M0.csv"
    if csv.exists():
        loaded = load_dispersion_curves([csv])[0][0]
        assert loaded.mode == Mode("M", 0)
        assert loaded.vs_err is not None


def test_cli_parses_and_reports_errors(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "evaluate",
                "--model",
                str(tmp_path / "missing.onnx"),
                "--benchmark",
                "b",
                "--out",
                "o",
            ]
        )
