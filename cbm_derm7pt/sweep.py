"""Small hyper-parameter sweep for Part 1. Selection uses the VALIDATION split only.

Runs run_part1.py for every combination in GRID and collects the validation metrics.

Usage:
    python sweep.py --data-dir ../release_v0 --features features/resnet50_derm.npz
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import pandas as pd

GRID = {
    "lr": [1e-3, 1e-4],
    "weight-decay": [1e-4, 1e-2],
    "hidden-dim": [0, 256],
    "dropout": [0.3, 0.6],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    feat_tag = "_".join(Path(p).stem for p in args.features)
    rows = []
    combos = list(itertools.product(*GRID.values()))
    for i, values in enumerate(combos, 1):
        params = dict(zip(GRID, values))
        run_name = f"sweep_{feat_tag}_" + "_".join(f"{k.replace('-', '')}{v}" for k, v in params.items())
        print(f"[{i}/{len(combos)}] {params}")
        cmd = [sys.executable, "run_part1.py", "--data-dir", args.data_dir, "--features", *args.features,
               "--run-name", run_name, "--seed", str(args.seed), "--n-random-orders", "5", "--no-plots"]
        for k, v in params.items():
            cmd += [f"--{k}", str(v)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

        out = Path("results") / run_name
        task = pd.read_csv(out / "task_metrics.csv")
        conc = pd.read_csv(out / "concept_metrics.csv")
        for model in ["independent", "sequential", "joint", "black_box"]:
            t = task[(task.split == "valid") & (task.model == model)].iloc[0]
            row = {**params, "model": model, "val_balanced_accuracy": t.balanced_accuracy,
                   "val_macro_auc": t.macro_auc_ovr, "run_name": run_name}
            c = conc[(conc.split == "valid") & (conc.model == model) & (conc.concept == "MEAN")]
            row["val_concept_balanced_accuracy"] = c.balanced_accuracy.iloc[0] if len(c) else float("nan")
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(Path("results") / f"sweep_{feat_tag}.csv", index=False)
    print("\nBest configuration per model (by validation balanced accuracy):")
    best = df.loc[df.groupby("model").val_balanced_accuracy.idxmax()]
    print(best.drop(columns="run_name").round(3).to_string(index=False))
    print(f"\nFull table: results/sweep_{feat_tag}.csv")


if __name__ == "__main__":
    main()
