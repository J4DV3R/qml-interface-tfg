"""Multi-seed benchmark of variational QCNN training (GD, Adam, SPSA, sequential).

Every (encoding, optimizer, initialization seed) combination is trained with
validation-only checkpoint selection. Unless ``--screening`` is given, test is
evaluated exactly once per run after the checkpoint is selected, with the same
shots/noise as training, plus an exact noise-free diagnostic of the same
checkpoint (never used for any selection). One JSON is written per run, plus a
``manifest.json`` with the configuration, provenance, data fingerprints and
classical/majority baselines. Outputs are written to a new directory only.

Example (tuning stage, no test):
    qml_env\\Scripts\\python.exe -m experiments.run_variational_benchmark ^
        --dataset breast-cancer --optimizers adam gd spsa sequential ^
        --init-seeds 101 102 103 --max-steps 60 --screening ^
        --output-dir results/variational/tuning-bc-exact
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.datasets import load_binary_breast_cancer_splits, load_binary_iris_splits
from src.fast_qcnn import FastQCNNConfig
from src.qcnn_engine import initialize_weights
from src.variational_training import (
    OPTIMIZERS,
    FastEvaluator,
    QiboEvaluator,
    VariationalTrainingConfig,
    binary_metrics,
    mse,
    train_variational_qcnn,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_RESULTS = REPOSITORY_ROOT / "results" / "initial-hybrid-adam"
SCHEMA_VERSION = 1


def _parse_shots(value: str) -> int | None:
    if value.strip().lower() in {"exact", "none"}:
        return None
    shots = int(value)
    if shots < 1:
        raise argparse.ArgumentTypeError("shots must be positive or 'exact'.")
    return shots


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPOSITORY_ROOT, check=True, capture_output=True
        ).stdout.decode("utf-8", errors="replace")
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_provenance(argv: list[str]) -> dict[str, Any]:
    """Revision, dirty state and a content hash of uncommitted code changes."""
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    diff = _git("diff", "HEAD", "--binary")
    untracked = []
    for line in (status or "").splitlines():
        if line.startswith("?? "):
            path = line[3:].strip()
            if path.split("/")[0] in {"src", "experiments", "tests"} and path.endswith(".py"):
                content = (REPOSITORY_ROOT / path).read_bytes()
                untracked.append({"path": path, "sha256": hashlib.sha256(content).hexdigest()})
    packages = {}
    for name in ("numpy", "scipy", "qibo", "scikit-learn", "torch"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "git_revision": (_git("rev-parse", "HEAD") or "").strip() or None,
        "working_tree_dirty": None if status is None else bool(status.strip()),
        "working_tree_status": (status or "").splitlines(),
        "uncommitted_diff_sha256": None if diff is None else hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "untracked_code_sha256": untracked,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "command_argv": [sys.executable, "-m", "experiments.run_variational_benchmark", *argv],
    }


def _fingerprint(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.round(array, 12), dtype=np.float64).tobytes()).hexdigest()


def load_partitions(dataset: str, seed: int, n_components: int):
    if dataset == "iris":
        splits = load_binary_iris_splits(seed=seed)
    else:
        splits = load_binary_breast_cancer_splits(seed=seed, n_components=n_components)

    def arr(tensor, flat=False):
        value = tensor.detach().cpu().numpy().astype(float)
        return value.reshape(-1) if flat else value

    data = {
        "train": (arr(splits.train_features), arr(splits.train_targets, True)),
        "validation": (arr(splits.validation_features), arr(splits.validation_targets, True)),
        "test": (arr(splits.test_features), arr(splits.test_targets, True)),
    }
    summary = {
        name: {
            "samples": int(len(y)),
            "positives": int(np.sum(y == 1)),
            "negatives": int(np.sum(y == -1)),
            "features_sha256": _fingerprint(x),
            "targets_sha256": _fingerprint(y),
        }
        for name, (x, y) in data.items()
    }
    return data, summary


def baselines(data, seed: int, include_test: bool) -> dict[str, Any]:
    train_x, train_y = data["train"]
    majority = 1.0 if np.sum(train_y == 1) >= np.sum(train_y == -1) else -1.0
    logistic = LogisticRegression(random_state=seed, max_iter=1000).fit(train_x, train_y)
    out: dict[str, Any] = {
        "majority_class": {"label": majority, "fitted_on": "train"},
        "logistic_regression": {"C": 1.0, "fitted_on": "train", "features": "same normalized features as the QCNN"},
    }
    partitions = ["validation"] + (["test"] if include_test else [])
    for name in partitions:
        x, y = data[name]
        out["majority_class"][name] = binary_metrics(np.full(len(y), majority), y)
        out["logistic_regression"][name] = binary_metrics(logistic.predict(x), y)
    return out


def make_evaluator(args, encoding: str, exact_diagnostic: bool = False):
    p = 0.0 if exact_diagnostic else args.noise
    r = 0.0 if exact_diagnostic else args.readout_error
    if args.backend == "qibo" and not exact_diagnostic:
        return QiboEvaluator(4, encoding, depolarizing_probability=p, readout_error=r)
    return FastEvaluator(
        FastQCNNConfig(4, 1, encoding, depolarizing_probability=p, readout_error=r)
    )


def training_config(args, optimizer: str) -> VariationalTrainingConfig:
    learning_rate = {"gd": args.gd_lr, "adam": args.adam_lr}.get(optimizer, 0.05)
    return VariationalTrainingConfig(
        optimizer=optimizer,
        max_steps=args.max_steps,
        learning_rate=learning_rate,
        shots=args.shots,
        patience=args.patience,
        max_optimization_circuits=args.max_optimization_circuits,
        evaluate_train_every=args.evaluate_train_every,
        spsa_a=args.spsa_a,
        spsa_c=args.spsa_c,
        spsa_stability=args.spsa_stability,
        sequential_grid_points=args.sequential_grid_points,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=("iris", "breast-cancer"), default="breast-cancer")
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--encodings", nargs="+", choices=("angle", "phase"), default=["angle"])
    parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS, default=list(OPTIMIZERS))
    parser.add_argument("--init-seeds", nargs="+", type=int, default=[19, 23, 29, 31, 37])
    parser.add_argument("--init-scale", type=float, default=0.25,
                        help="initial weights ~ U(-scale, scale); 0.25 matches the historical runs")
    parser.add_argument("--execution-seed", type=int, default=101)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-optimization-circuits", type=int, default=None,
                        help="optional equal circuit budget for all optimizers")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--evaluate-train-every", type=int, default=1)
    parser.add_argument("--shots", type=_parse_shots, default=None, help="'exact' (default) or a positive integer")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="local depolarizing probability after RY/RZ/CNOT (simulator sensitivity, not a calibration)")
    parser.add_argument("--readout-error", type=float, default=0.0, help="symmetric readout bit-flip probability")
    parser.add_argument("--backend", choices=("fast", "qibo"), default="fast")
    parser.add_argument("--gd-lr", type=float, default=0.5)
    parser.add_argument("--adam-lr", type=float, default=0.05)
    parser.add_argument("--spsa-a", type=float, default=0.2)
    parser.add_argument("--spsa-c", type=float, default=0.1)
    parser.add_argument("--spsa-stability", type=float, default=None,
                        help="SPSA stability constant A; default 0.1 * max_steps")
    parser.add_argument("--sequential-grid-points", type=int, default=2048)
    parser.add_argument("--screening", action="store_true",
                        help="never touch test: use for hyperparameter/budget tuning on validation")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory (relative paths are rooted at the repository)")
    return parser


def resolve_output_dir(path: Path) -> Path:
    resolved = (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()
    try:
        resolved.relative_to(FROZEN_RESULTS.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("output cannot be inside results/initial-hybrid-adam (frozen).")
    if resolved.exists():
        raise ValueError(f"output directory already exists; refusing to overwrite: {resolved}")
    return resolved


def run(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    try:
        output_dir = resolve_output_dir(args.output_dir)
        if args.n_components != 4:
            raise ValueError("the QCNN uses 4 qubits: --n-components must be 4.")
        if not 0.0 <= args.noise <= 1.0 or not 0.0 <= args.readout_error <= 0.5:
            raise ValueError("noise must be in [0, 1] and readout error in [0, 0.5].")
        if args.backend == "qibo" and args.readout_error:
            raise ValueError("readout error is only modeled by the fast backend.")
        for optimizer in args.optimizers:
            training_config(args, optimizer)
    except ValueError as exc:
        parser.error(str(exc))

    data, partition_summary = load_partitions(args.dataset, args.dataset_seed, args.n_components)
    include_test = not args.screening
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "variational_qcnn_benchmark",
        "screening": args.screening,
        "created_unix": time.time(),
        "provenance": collect_provenance(arguments),
        "configuration": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "dataset": {
            "name": args.dataset,
            "seed": args.dataset_seed,
            "n_components": args.n_components if args.dataset == "breast-cancer" else None,
            "split_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "preprocessing": "stratified split; min-max (and for Breast Cancer standardization+PCA) fitted on train only",
            "partitions": partition_summary,
        },
        "noise_model": {
            "depolarizing_probability": args.noise,
            "placement": "independent single-qubit depolarizing channel on each qubit touched by RY, RZ, CNOT (encoding included, H excluded)",
            "readout_error": args.readout_error,
            "simulation": "exact density matrix; shots are independent binomial samples",
            "scope": "simulator sensitivity model; not a hardware calibration",
        },
        "classical_quantum_boundary": (
            "Circuits produce Z expectations (exact or from shots); the host computes "
            "losses, parameter-shift combinations or reconstructions, optimizer "
            "updates, validation checkpointing and test metrics. This is the standard "
            "variational (hybrid) execution model for current hardware."
        ),
        "baselines": baselines(data, args.dataset_seed, include_test),
        "status": "started",
        "runs_planned": [
            f"run-{encoding}-{optimizer}-init{seed}.json"
            for encoding in args.encodings
            for optimizer in args.optimizers
            for seed in args.init_seeds
        ],
        "runs": [],
    }
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    train_x, train_y = data["train"]
    val_x, val_y = data["validation"]
    test_x, test_y = data["test"]

    for encoding in args.encodings:
        for optimizer in args.optimizers:
            config = training_config(args, optimizer)
            for init_seed in args.init_seeds:
                rng = np.random.default_rng(np.random.SeedSequence([args.execution_seed, init_seed]))
                initial = initialize_weights(4, seed=init_seed, low=-args.init_scale, high=args.init_scale)
                evaluator = make_evaluator(args, encoding)
                result = train_variational_qcnn(
                    train_x, train_y, val_x, val_y, initial, evaluator, config, rng
                )
                payload: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "condition": {
                        "dataset": args.dataset,
                        "encoding": encoding,
                        "optimizer": optimizer,
                        "shots": args.shots,
                        "noise": args.noise,
                        "readout_error": args.readout_error,
                        "backend": args.backend,
                    },
                    "seeds": {
                        "dataset": args.dataset_seed,
                        "initialization": init_seed,
                        "execution": args.execution_seed,
                        "execution_stream": "numpy SeedSequence([execution_seed, initialization_seed])",
                    },
                    "training_config": asdict(config),
                    "hyperparameters": config.effective_hyperparameters(),
                    "training": result.to_dict(),
                }
                if include_test:
                    measured = evaluator.predict(result.best_weights, test_x, shots=args.shots, rng=rng)
                    exact = make_evaluator(args, encoding, exact_diagnostic=True).predict(
                        result.best_weights, test_x, shots=None, rng=None
                    )
                    payload["test"] = {
                        "evaluated_once_after_checkpoint_selection": True,
                        "checkpoint_step": result.best_step,
                        "circuits": len(test_x),
                        "mse_loss": mse(measured, test_y),
                        "metrics": binary_metrics(measured, test_y),
                        "exact_noiseless_diagnostic": {
                            "note": "same checkpoint, exact noise-free expectations; diagnostic only, never used for selection",
                            "mse_loss": mse(exact, test_y),
                            "metrics": binary_metrics(exact, test_y),
                        },
                    }
                else:
                    payload["test"] = {"evaluated": False, "reason": "screening"}
                name = f"run-{encoding}-{optimizer}-init{init_seed}.json"
                with (output_dir / name).open("x", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, allow_nan=False)
                manifest["runs"].append(name)
                summary = payload["test"].get("metrics", {}).get("accuracy")
                print(
                    f"{name}: best_step={result.best_step} val_loss={result.best_validation_loss:.4f}"
                    + (f" test_acc={summary:.4f}" if summary is not None else "")
                    + f" ({result.elapsed_seconds:.1f}s)",
                    flush=True,
                )
    # Rewrites only the manifest this invocation created in its own new directory.
    manifest["status"] = "completed"
    manifest["completed_unix"] = time.time()
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    print(f"Wrote {len(manifest['runs'])} runs and manifest.json to {output_dir}")
    return output_dir


if __name__ == "__main__":
    run()
