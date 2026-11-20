"""Robustness analysis for Part 1.

1. Seeds:  the final configuration is re-run with 5 seeds -> mean +- std of every metric.
2. Lambda: the joint CBM is re-run with several concept-loss weights (lambda), 5 seeds each,
           to test whether the concept leakage depends on lambda.

Runs that already exist in results/ are reused, so the script can be interrupted and restarted.

Usage:
    python robustness.py --data-dir ../release_v0 --features features/dinov2_vitb14_derm.npz
Outputs: results/robustness/
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SEEDS = [0, 1, 2, 3, 4]
LAMBDAS = [0.01, 0.1, 1.0, 10.0]
FINAL_LAMBDA = 1.0

# Final configuration (selected on the validation set by sweep.py)
CBM_ARGS = ["--lr", "1e-3", "--weight-decay", "1e-2", "--hidden-dim", "256", "--dropout", "0.3"]
BB_ARGS = ["--bb-lr", "1e-4", "--bb-weight-decay", "1e-4", "--bb-hidden-dim", "0", "--bb-dropout", "0.6"]

REGIMES = ["independent", "sequential", "joint"]
COLOURS = {"independent": "#2a78b5", "sequential": "#e08a1e", "joint": "#3a9a5b", "black_box": "#7a7a7a"}
TASK_METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc_ovr",
                "melanoma_sensitivity", "melanoma_specificity"]


def run_name(lam, seed):
    return f"final_lambda{lam:g}_seed{seed}"


def run(args, lam, seed):
    out = Path("results") / run_name(lam, seed)
    if (out / "task_metrics.csv").exists():
        print(f"  reusing {out}")
        return out
    print(f"  training lambda={lam:g}, seed={seed}")
    cmd = [sys.executable, "run_part1.py", "--data-dir", args.data_dir, "--features", *args.features,
           "--run-name", run_name(lam, seed), "--seed", str(seed), "--joint-lambda", str(lam),
           "--no-plots", *CBM_ARGS, *BB_ARGS]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return out


def collect(runs, filename):
    frames = []
    for (lam, seed), out in runs.items():
        frames.append(pd.read_csv(out / filename).assign(joint_lambda=lam, seed=seed))
    return pd.concat(frames, ignore_index=True)


def mean_std(df, keys, metrics):
    agg = df.groupby(keys)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def fmt(df, keys, metrics):
    """Readable 'mean ± std' table."""
    out = df[keys].copy()
    for m in metrics:
        out[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(df[f"{m}_mean"], df[f"{m}_std"])]
    return out


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

def seed_analysis(runs, out_dir):
    seed_runs = {k: v for k, v in runs.items() if k[0] == FINAL_LAMBDA}

    task = collect(seed_runs, "task_metrics.csv")
    task = mean_std(task[task.split == "test"], ["model"], TASK_METRICS)
    order = REGIMES + [f"{r}_true_concepts" for r in REGIMES] + ["black_box"]
    task = task.set_index("model").loc[order].reset_index()
    task.to_csv(out_dir / "seeds_task_metrics.csv", index=False)

    conc = collect(seed_runs, "concept_metrics.csv")
    conc = mean_std(conc[conc.split == "test"], ["model", "concept"],
                    ["accuracy", "balanced_accuracy", "macro_f1"])
    conc.to_csv(out_dir / "seeds_concept_metrics.csv", index=False)

    rule = collect(seed_runs, "seven_point_rule.csv")
    rule = mean_std(rule[rule.split == "test"], ["concepts"],
                    ["melanoma_sensitivity", "melanoma_specificity", "melanoma_auc"])
    rule.to_csv(out_dir / "seeds_seven_point_rule.csv", index=False)

    interv = collect(seed_runs, "interventions.csv")
    interv = mean_std(interv[interv.split == "test"], ["model", "order", "k"], ["accuracy", "balanced_accuracy"])
    interv.to_csv(out_dir / "seeds_interventions.csv", index=False)

    print(f"\n=== Test set, {len(SEEDS)} seeds (mean ± std) ===")
    print(fmt(task, ["model"], ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc_ovr",
                                "melanoma_sensitivity"]).to_string(index=False))
    print("\nMean concept performance")
    m = conc[conc.concept == "MEAN"]
    print(fmt(m, ["model"], ["accuracy", "balanced_accuracy", "macro_f1"]).to_string(index=False))
    print("\n7-point rule (melanoma)")
    print(fmt(rule, ["concepts"], ["melanoma_sensitivity", "melanoma_specificity", "melanoma_auc"]).to_string(index=False))

    # Task performance with error bars
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc_ovr", "melanoma_sensitivity"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    width = 0.2
    for i, model in enumerate(REGIMES + ["black_box"]):
        d = task[task.model == model].iloc[0]
        ax.bar([j + (i - 1.5) * width for j in range(len(metrics))], [d[f"{m}_mean"] for m in metrics], width,
               yerr=[d[f"{m}_std"] for m in metrics], capsize=2, color=COLOURS[model], label=model)
    ax.set_xticks(range(len(metrics)), [m.replace("_", " ") for m in metrics])
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, fontsize=8)
    ax.set_title(f"Diagnosis performance, test set (mean ± std over {len(SEEDS)} seeds)")
    fig.tight_layout()
    fig.savefig(out_dir / "seeds_task_metrics.png", dpi=200)
    plt.close(fig)

    # Intervention curves with error bands
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, metric in zip(axes, ["accuracy", "balanced_accuracy"]):
        for regime in REGIMES:
            for order, style in [("random", "-"), ("uncertain", "--")]:
                d = interv[(interv.model == regime) & (interv.order == order)].sort_values("k")
                ax.plot(d.k, d[f"{metric}_mean"], style, marker="o", ms=3, color=COLOURS[regime],
                        label=f"{regime} ({order})")
                ax.fill_between(d.k, d[f"{metric}_mean"] - d[f"{metric}_std"],
                                d[f"{metric}_mean"] + d[f"{metric}_std"], color=COLOURS[regime], alpha=0.10)
        ax.set_xlabel("Number of concept groups intervened on")
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Concept interventions, test set (mean ± std over {len(SEEDS)} seeds)")
    fig.tight_layout()
    fig.savefig(out_dir / "seeds_interventions.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Lambda sweep for the joint CBM
# ---------------------------------------------------------------------------

def lambda_analysis(runs, out_dir):
    task = collect(runs, "task_metrics.csv")
    task = task[task.split == "test"]
    joint = task[task.model == "joint"][["joint_lambda", "seed", "accuracy", "balanced_accuracy"]]
    oracle = task[task.model == "joint_true_concepts"][["joint_lambda", "seed", "balanced_accuracy"]]
    oracle = oracle.rename(columns={"balanced_accuracy": "balanced_accuracy_true_concepts"})

    conc = collect(runs, "concept_metrics.csv")
    conc = conc[(conc.split == "test") & (conc.model == "joint") & (conc.concept == "MEAN")]
    conc = conc[["joint_lambda", "seed", "balanced_accuracy"]].rename(
        columns={"balanced_accuracy": "concept_balanced_accuracy"})

    df = joint.merge(oracle, on=["joint_lambda", "seed"]).merge(conc, on=["joint_lambda", "seed"])
    # Leakage indicator: how much the diagnosis gets WORSE when the true concepts are given.
    df["intervention_gain"] = df["balanced_accuracy_true_concepts"] - df["balanced_accuracy"]
    metrics = ["accuracy", "balanced_accuracy", "balanced_accuracy_true_concepts",
               "concept_balanced_accuracy", "intervention_gain"]
    agg = mean_std(df, ["joint_lambda"], metrics)
    agg.to_csv(out_dir / "lambda_sweep.csv", index=False)

    # Reference: independent CBM (does not depend on lambda) averaged over all runs
    ind = task[task.model == "independent"].balanced_accuracy.mean()
    ind_true = task[task.model == "independent_true_concepts"].balanced_accuracy.mean()

    print(f"\n=== Joint CBM vs lambda, test set ({len(SEEDS)} seeds, mean ± std) ===")
    print(fmt(agg, ["joint_lambda"], metrics).to_string(index=False))
    print(f"Reference independent CBM: balanced acc {ind:.3f}, with true concepts {ind_true:.3f}")

    interv = collect(runs, "interventions.csv")
    interv = interv[(interv.split == "test") & (interv.model == "joint") & (interv.order == "random")]
    interv = mean_std(interv, ["joint_lambda", "k"], ["balanced_accuracy"])
    interv.to_csv(out_dir / "lambda_interventions.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for col, label, colour in [("balanced_accuracy", "diagnosis (predicted concepts)", "#3a9a5b"),
                               ("balanced_accuracy_true_concepts", "diagnosis (true concepts)", "#9b59b6"),
                               ("concept_balanced_accuracy", "concepts (mean)", "#c0392b")]:
        ax.errorbar(agg.joint_lambda, agg[f"{col}_mean"], yerr=agg[f"{col}_std"], marker="o",
                    capsize=3, color=colour, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("lambda (weight of the concept loss)")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Joint CBM: task vs concept trade-off")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    cmap = plt.get_cmap("viridis")
    for i, lam in enumerate(LAMBDAS):
        d = interv[interv.joint_lambda == lam].sort_values("k")
        colour = cmap(i / max(len(LAMBDAS) - 1, 1))
        ax.plot(d.k, d.balanced_accuracy_mean, marker="o", ms=3, color=colour, label=f"lambda = {lam:g}")
        ax.fill_between(d.k, d.balanced_accuracy_mean - d.balanced_accuracy_std,
                        d.balanced_accuracy_mean + d.balanced_accuracy_std, color=colour, alpha=0.12)
    ax.set_xlabel("Number of concept groups intervened on (random order)")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Joint CBM: interventions vs lambda")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "lambda_sweep.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    args = parser.parse_args()

    runs = {}
    for lam in LAMBDAS:
        for seed in SEEDS:
            runs[(lam, seed)] = run(args, lam, seed)

    out_dir = Path("results") / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_analysis(runs, out_dir)
    lambda_analysis(runs, out_dir)
    print(f"\nTables and figures written to {out_dir}")


if __name__ == "__main__":
    main()
