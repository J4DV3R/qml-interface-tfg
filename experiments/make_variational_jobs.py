"""Write a jobs file (one CLI argument line per job) for Slurm job arrays.

Each line holds the arguments of one ``experiments.run_variational_benchmark``
invocation (one dataset partition x optimizer x encoding, all initialization
seeds), exactly as the phase-C runs were organised. ``slurm/mn5_gpp_array.sbatch``
reads line ``$SLURM_ARRAY_TASK_ID`` of the file.

Example (phase-C grid):
    qml_env/Scripts/python.exe -B -m experiments.make_variational_jobs \
        --datasets breast-cancer iris --dataset-seeds 1 2 3 4 5 6 7 8 9 10 \
        --output-root results/variational/C-partitions --jobs-file slurm/jobs/phaseC.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

HYPERPARAMETERS = "--gd-lr 0.5 --adam-lr 0.1 --spsa-a 1.6 --spsa-c 0.1 --spsa-stability 6"


def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", default=["breast-cancer", "iris"])
    parser.add_argument("--dataset-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--optimizers", nargs="+", default=["gd", "adam", "spsa", "sequential"])
    parser.add_argument("--encodings", nargs="+", default=["angle", "phase"])
    parser.add_argument("--init-seeds", nargs="+", type=int, default=[19, 23, 29, 31, 37, 41, 43, 47])
    parser.add_argument("--execution-seed", type=int, default=101)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--extra", type=str, default="", help="extra arguments appended to every job")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs-file", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.jobs_file.exists():
        raise SystemExit(f"refusing to overwrite {args.jobs_file}")

    lines = []
    for dataset in args.datasets:
        for seed in args.dataset_seeds:
            for optimizer in args.optimizers:
                for encoding in args.encodings:
                    out = args.output_root / dataset / f"ds{seed}" / f"{optimizer}-{encoding}"
                    lines.append(" ".join([
                        f"--dataset {dataset} --dataset-seed {seed}",
                        f"--encodings {encoding} --optimizers {optimizer}",
                        "--init-seeds " + " ".join(map(str, args.init_seeds)),
                        f"--execution-seed {args.execution_seed} --max-steps {args.max_steps}",
                        HYPERPARAMETERS,
                        args.extra,
                        f"--output-dir {out.as_posix()}",
                    ]).replace("  ", " ").strip())
    args.jobs_file.parent.mkdir(parents=True, exist_ok=True)
    args.jobs_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} jobs to {args.jobs_file} (use --array=1-{len(lines)})")
    return lines


if __name__ == "__main__":
    main()
