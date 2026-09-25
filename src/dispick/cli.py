"""The `dispick` command.

dispick bank build --out data/bank.h5 --models 50000
dispick data shard --bank data/bank.h5 --out data/validation.h5 --count 4096 --seed 1
dispick data benchmark --bank data/bank.h5 --out data/benchmark.h5 --count 2000 --seed 2
dispick data preview --bank data/bank.h5 --out preview.png
dispick train --config configs/train.yaml [--set optim.steps=1000 ...]
dispick export --checkpoint runs/base/best.pt --out models/dispick.onnx
dispick evaluate --model models/dispick.onnx --benchmark data/benchmark.h5 --out report/
dispick pick --model models/dispick.onnx xmid_*/DispersionImage_0000.hdf5 --out picks/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger("dispick")


def _synthesis(path: Path | None) -> object:
    from dispick.synthesis.config import SynthesisConfig

    if path is None:
        return SynthesisConfig()
    return SynthesisConfig.model_validate(yaml.safe_load(path.read_text()) or {})


def _bank_build(args: argparse.Namespace) -> None:
    from dispick.physics.bank import BankConfig, build_bank

    raw = yaml.safe_load(args.config.read_text()) if args.config else {}
    raw = raw or {}
    if args.models is not None:
        raw["n_models"] = args.models
    if args.seed is not None:
        raw["seed"] = args.seed
    config = BankConfig.model_validate(raw)
    path = build_bank(config, args.out, workers=args.workers)
    print(path)


def _spec(args: argparse.Namespace) -> object:
    from dispick.data.generate import DatasetSpec

    return DatasetSpec(
        bank=args.bank,
        synthesis=_synthesis(args.synthesis),  # pyright: ignore[reportArgumentType]
        seed=args.seed,
        grid=(args.grid[0], args.grid[1]),
        n_modes=args.modes,
    )


def _data_shard(args: argparse.Namespace) -> None:
    from dispick.data.generate import build_shard

    print(build_shard(_spec(args), args.out, args.count, args.workers))  # pyright: ignore[reportArgumentType]


def _data_benchmark(args: argparse.Namespace) -> None:
    from dispick.data.generate import build_benchmark

    print(build_benchmark(_spec(args), args.out, args.count, args.workers))  # pyright: ignore[reportArgumentType]


def _data_preview(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dispick.physics.bank import ModalBank
    from dispick.synthesis.generator import SyntheticGenerator

    generator = SyntheticGenerator(ModalBank(args.bank), _synthesis(args.synthesis))  # pyright: ignore[reportArgumentType]
    rng = np.random.default_rng(args.seed)
    samples = [generator.sample(rng) for _ in range(args.count)]
    columns = 6
    rows = -(-len(samples) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.6 * rows), squeeze=False)
    for ax, sample in zip(axes.ravel(), samples, strict=False):
        fs, vs = sample.frequencies, sample.velocities
        ax.imshow(
            sample.fv_map.T, origin="lower", aspect="auto", cmap="turbo", vmin=0, vmax=1,
            extent=(float(fs[0]), float(fs[-1]), float(vs[0]), float(vs[-1])),
            interpolation="nearest",
        )  # fmt: skip
        for mode in range(1, sample.curves.shape[0]):
            ax.plot(fs, sample.curves[mode], "w--", lw=0.6, alpha=0.7)
        ax.plot(fs, sample.curves[0], "w-", lw=1.0)
        ax.plot(fs[sample.visible], sample.curves[0][sample.visible], "m.", ms=3)
        ax.set_ylim(float(vs[0]), float(vs[-1]))
        info, labels = sample.info, sample.labels
        ax.set_title(
            f"{info['scenario']} {info['kind']} {info['family']} N={info['n_receivers']}\n"
            f"pickable={labels.pickable} q={labels.quality:.2f} hm={labels.higher_mode_share:.2f}",
            fontsize=7,
        )
    for ax in axes.ravel()[len(samples) :]:
        ax.axis("off")
    figure.tight_layout()
    figure.savefig(args.out, dpi=70)
    print(args.out)


def _train(args: argparse.Namespace) -> None:
    from dispick.training.config import TrainConfig
    from dispick.training.trainer import train

    overrides: dict[str, object] = {}
    for item in args.set or []:
        key, _, value = item.partition("=")
        overrides[key] = yaml.safe_load(value)
    config = TrainConfig.from_yaml(args.config, overrides)
    print(train(config, resume=not args.no_resume))


def _export(args: argparse.Namespace) -> None:
    from dispick.export import export_onnx

    print(export_onnx(args.checkpoint, args.out))


def _evaluate(args: argparse.Namespace) -> None:
    from dispick.data.shards import read_benchmark
    from dispick.evaluation.baselines import maximum_method
    from dispick.evaluation.benchmark import (
        Method,
        picker_method,
        precomputed_method,
        write_report,
    )
    from dispick.evaluation.benchmark import evaluate as run
    from dispick.inference.picker import Picker, PickSettings

    picker = Picker.load(args.model, device=args.device)
    methods: dict[str, Method] = {
        "dispick": picker_method(picker, PickSettings(threshold=args.threshold, zoom=args.zoom))
    }
    if args.baselines:
        methods["maximum"] = maximum_method
    for item in args.external or []:
        name, _, path = item.partition("=")
        methods[name] = precomputed_method(Path(path))
    samples = list(read_benchmark(args.benchmark))
    if args.limit:
        samples = samples[: args.limit]
    reports = write_report(run(samples, methods), args.out)
    print(json.dumps({name: r["overall"] for name, r in reports.items()}, indent=2))


def _pick(args: argparse.Namespace) -> None:
    from dispick.io import pick_files

    for line in pick_files(args.images, args.out, args.model, args.threshold, args.figure):
        print(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dispick", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    bank = commands.add_parser("bank", help="the modal bank").add_subparsers(
        dest="action", required=True
    )
    build = bank.add_parser("build", help="compute a bank of random earth models and their modes")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--config", type=Path, help="a BankConfig YAML")
    build.add_argument("--models", type=int)
    build.add_argument("--seed", type=int)
    build.add_argument("--workers", type=int)
    build.set_defaults(run=_bank_build)

    data = commands.add_parser("data", help="fixed datasets").add_subparsers(
        dest="action", required=True
    )
    for name, function, help_text in (
        ("shard", _data_shard, "training/validation examples on the canonical grid"),
        ("benchmark", _data_benchmark, "whole samples at their own axes, for `evaluate`"),
        ("preview", _data_preview, "a figure of synthetic images with their truth"),
    ):
        sub = data.add_parser(name, help=help_text)
        sub.add_argument("--bank", type=Path, required=True)
        sub.add_argument("--out", type=Path, required=True)
        sub.add_argument("--count", type=int, default=24 if name == "preview" else 4096)
        sub.add_argument("--seed", type=int, default=1)
        sub.add_argument("--synthesis", type=Path, help="a SynthesisConfig YAML")
        sub.add_argument("--grid", type=int, nargs=2, default=(256, 256))
        sub.add_argument("--modes", type=int, default=4)
        sub.add_argument("--workers", type=int, default=1)
        sub.set_defaults(run=function)

    train = commands.add_parser("train", help="train the network")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config key")
    train.add_argument("--no-resume", action="store_true")
    train.set_defaults(run=_train)

    export = commands.add_parser("export", help="a checkpoint to ONNX and its card")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    export.set_defaults(run=_export)

    evaluate = commands.add_parser("evaluate", help="score a model on a benchmark")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--benchmark", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--threshold", type=float, default=0.5)
    evaluate.add_argument("--zoom", action="store_true")
    evaluate.add_argument("--baselines", action="store_true")
    evaluate.add_argument(
        "--external", action="append", metavar="NAME=PICKS.npz", help="picks made elsewhere"
    )
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--device", default="cpu")
    evaluate.set_defaults(run=_evaluate)

    pick = commands.add_parser("pick", help="pick sigpipe/PAC dispersion image files")
    pick.add_argument("images", type=Path, nargs="+")
    pick.add_argument("--model", type=Path)
    pick.add_argument("--out", type=Path, required=True)
    pick.add_argument("--threshold", type=float, default=0.5)
    pick.add_argument("--figure", action="store_true")
    pick.set_defaults(run=_pick)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        args.run(args)
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
