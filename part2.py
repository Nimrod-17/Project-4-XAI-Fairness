"""Part 2: validating the explanations of the three CBM regimes.

Runs entirely on the saved Part 1 outputs (no retraining, no torch needed):
    results/final_lambda<L>_seed<S>/predictions.csv          per-patient concept probabilities
    results/final_lambda<L>_seed<S>/label_predictor_weights.csv   weights + bias of the linear f

Because f is a single linear layer, f(c) = softmax(W c + b) can be recomputed exactly
for any (corrected) bottleneck c, so every intervention can be simulated from these files.

Use case A - clinician-in-the-loop correction
    Ideal:    clinician + AI beats AI alone, with <= 3 corrections per case.
    Testable: correcting the k most uncertain concept groups (k <= 3) with the annotated value
              raises balanced accuracy, also when the "clinician" is wrong with probability p.
    Proxies:  annotations = clinician; most-uncertain-first = clinician's choice;
              number of corrections = time.

Use case B - knowing when not to trust the model
    Ideal:    clinicians shown the concepts reject wrong predictions more often.
    Testable: scores computed from the explanation detect wrong diagnoses (AUROC) better than
              the model's own confidence, and add information on top of it.
    Proxies:  an algorithmic suspicion score = a clinician noticing something is off;
              AUROC on derm7pt test = less over-reliance in practice.

Usage (from the repository root):
    python part2.py                       # lambda = 1, seeds 0-4, test split
    python part2.py --joint-lambda 10     # joint CBM with another lambda
    python part2.py --export-study        # 20 cases for the optional group study
Outputs go to cbm_derm7pt/results/part2_lambda<L>/.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from cbm_derm7pt.data import CONCEPT_GROUPS, DIAGNOSIS_CLASSES, GROUP_SIZES, group_slices, seven_point_score

REGIMES = ["independent", "sequential", "joint"]
SLICES = group_slices()
N_GROUPS = len(CONCEPT_GROUPS)
MEL = DIAGNOSIS_CLASSES.index("MEL")
COLOURS = {"independent": "#2e6fa7", "sequential": "#e08a1e", "joint": "#3a9a5b"}


# ---------------------------------------------------------------------------
# Loading the saved Part 1 outputs
# ---------------------------------------------------------------------------

def load_run(run_dir, regime, split):
    """Bottleneck, true labels and the linear f of one regime in one saved run."""
    pred = pd.read_csv(run_dir / "predictions.csv")
    pred = pred[(pred.model == regime) & (pred.split == split)].sort_values("row_index")
    weights = pd.read_csv(run_dir / "label_predictor_weights.csv")
    weights = weights[weights.model == regime].set_index("class").loc[DIAGNOSIS_CLASSES]

    unit_cols = [f"{g.abbrev}:{c}" for g in CONCEPT_GROUPS for c in g.classes]
    c_true = np.stack([pred[f"true_{g.abbrev}"].map(list(g.classes).index).to_numpy()
                       for g in CONCEPT_GROUPS], axis=1)
    run = {
        "row_index": pred.row_index.to_numpy(),
        "y": pred.y_true.map(DIAGNOSIS_CLASSES.index).to_numpy(),
        "bottleneck": pred[[f"c_{u}" for u in unit_cols]].to_numpy(dtype=np.float64),
        "c_true": c_true,
        "W": weights[unit_cols].to_numpy(dtype=np.float64),
        "b": weights["bias"].to_numpy(dtype=np.float64),
    }
    # Sanity check: recomputing f must reproduce the saved probabilities.
    saved = pred[[f"p_{k}" for k in DIAGNOSIS_CLASSES]].to_numpy()
    assert np.allclose(f_probs(run, run["bottleneck"]), saved, atol=1e-4), "f does not match saved run"
    return run


def f_probs(run, bottleneck):
    z = bottleneck @ run["W"].T + run["b"]
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def group_entropy(bottleneck):
    """Normalised entropy per concept group, (N, 7) in [0, 1] (as in evaluation.py)."""
    ent = []
    for sl in SLICES:
        p = np.clip(bottleneck[:, sl], 1e-12, 1.0)
        ent.append(-(p * np.log(p)).sum(1) / np.log(sl.stop - sl.start))
    return np.stack(ent, axis=1)


def argmax_concepts(bottleneck):
    return np.stack([bottleneck[:, sl].argmax(1) for sl in SLICES], axis=1)


def to_onehot(labels):
    out = np.zeros((labels.shape[0], sum(GROUP_SIZES)))
    for g, sl in enumerate(SLICES):
        out[np.arange(len(labels)), sl.start + labels[:, g]] = 1.0
    return out


def balanced_accuracy(y, y_hat, n_classes=len(DIAGNOSIS_CLASSES)):
    """Fast balanced accuracy (mean per-class recall over classes present in y)."""
    total = np.bincount(y, minlength=n_classes)
    hit = np.bincount(y[y == y_hat], minlength=n_classes)
    present = total > 0
    return float((hit[present] / total[present]).mean())


def mean_std(values):
    return f"{np.mean(values):.3f} ± {np.std(values):.3f}"


# ---------------------------------------------------------------------------
# Use case A: correction by a (possibly noisy) clinician
# ---------------------------------------------------------------------------

def corrected_bottleneck(run, k, p_wrong, rng):
    """Replace the k most uncertain groups by the annotation; with probability p_wrong the
    'clinician' instead enters a uniformly chosen wrong class for that group."""
    bottleneck = run["bottleneck"].copy()
    if k == 0:
        return bottleneck
    order = np.argsort(-group_entropy(run["bottleneck"]), axis=1)[:, :k]
    n = len(bottleneck)
    for j in range(k):
        groups = order[:, j]
        entered = run["c_true"][np.arange(n), groups].copy()
        mistake = rng.random(n) < p_wrong
        for i in np.where(mistake)[0]:
            size = GROUP_SIZES[groups[i]]
            wrong = [c for c in range(size) if c != entered[i]]
            entered[i] = rng.choice(wrong)
        for i in range(n):
            sl = SLICES[groups[i]]
            bottleneck[i, sl] = 0.0
            bottleneck[i, sl.start + entered[i]] = 1.0
    return bottleneck


def use_case_a(runs, noise_levels, n_draws, seed=0):
    """Returns per-run predictions (for bootstrapping) and a tidy table of results."""
    rows, preds = [], {}
    for (regime, s), run in runs.items():
        base_pred = f_probs(run, run["bottleneck"]).argmax(1)
        for p in noise_levels:
            draws = 1 if p == 0 else n_draws
            rng = np.random.default_rng(seed + 1000 * s)
            for k in range(N_GROUPS + 1):
                for d in range(draws):
                    y_hat = f_probs(run, corrected_bottleneck(run, k, p, rng)).argmax(1)
                    preds[(regime, s, p, k, d)] = y_hat
                    base_ok, new_ok = base_pred == run["y"], y_hat == run["y"]
                    rows.append({
                        "regime": regime, "seed": s, "p_wrong": p, "k": k, "draw": d,
                        "balanced_accuracy": balanced_accuracy(run["y"], y_hat),
                        "accuracy": float(new_ok.mean()),
                        # harm: correct before correction, wrong after
                        "harm_rate": float((base_ok & ~new_ok).mean()),
                        "fix_rate": float((~base_ok & new_ok).mean()),
                    })
    return pd.DataFrame(rows), preds


# ---------------------------------------------------------------------------
# Use case B: does the explanation flag wrong diagnoses?
# ---------------------------------------------------------------------------

def suspicion_scores(run):
    """Per-patient scores; higher = more suspicious. Only 'confidence' ignores the explanation."""
    probs = f_probs(run, run["bottleneck"])
    pred_concepts = argmax_concepts(run["bottleneck"])
    hard_probs = f_probs(run, to_onehot(pred_concepts))
    rule_mel = seven_point_score(pred_concepts) >= 3
    model_mel = probs.argmax(1) == MEL
    return {
        "confidence (baseline)": 1.0 - probs.max(1),
        "concept uncertainty": group_entropy(run["bottleneck"]).mean(1),
        "7-pt rule disagrees": (rule_mel != model_mel).astype(float),
        "hard vs soft concepts": 0.5 * np.abs(hard_probs - probs).sum(1),  # total variation
    }


def use_case_b(runs, val_runs):
    """AUROC of each score for detecting wrong diagnoses, plus a 'beyond confidence' test:
    logistic regression on [confidence] vs [confidence + explanation scores], fitted on the
    validation split and evaluated on the test split."""
    rows, raw = [], {}
    for key, run in runs.items():
        wrong = (f_probs(run, run["bottleneck"]).argmax(1) != run["y"]).astype(int)
        scores = suspicion_scores(run)
        raw[key] = (wrong, scores)
        for name, sc in scores.items():
            rows.append({"regime": key[0], "seed": key[1], "score": name,
                         "auroc": roc_auc_score(wrong, sc), "error_rate": wrong.mean()})

        val = val_runs[key]
        val_wrong = (f_probs(val, val["bottleneck"]).argmax(1) != val["y"]).astype(int)
        val_scores = suspicion_scores(val)
        names = list(scores)
        for label, cols in [("LR: confidence only", names[:1]), ("LR: confidence + explanation", names)]:
            Xv = np.column_stack([val_scores[c] for c in cols])
            Xt = np.column_stack([scores[c] for c in cols])
            lr = LogisticRegression(max_iter=1000).fit(Xv, val_wrong)
            p = lr.predict_proba(Xt)[:, 1]
            raw[key][1][label] = p
            rows.append({"regime": key[0], "seed": key[1], "score": label,
                         "auroc": roc_auc_score(wrong, p), "error_rate": wrong.mean()})
    return pd.DataFrame(rows), raw


# ---------------------------------------------------------------------------
# Bootstrap over test patients (seed-averaged), for 95% intervals
# ---------------------------------------------------------------------------

def bootstrap_a(runs, preds, seeds, n_boot, noise_levels, n_draws, k=3, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    n = len(next(iter(runs.values()))["y"])
    rows = []
    for regime in REGIMES:
        for p in noise_levels:
            draws = 1 if p == 0 else n_draws
            gains = []
            for _ in range(n_boot):
                idx = rng.integers(0, n, n)
                per_seed = []
                for s in seeds:
                    y = runs[(regime, s)]["y"][idx]
                    base = balanced_accuracy(y, preds[(regime, s, p, 0, 0)][idx])
                    after = np.mean([balanced_accuracy(y, preds[(regime, s, p, k, d)][idx])
                                     for d in range(draws)])
                    per_seed.append(after - base)
                gains.append(np.mean(per_seed))
            lo, hi = np.percentile(gains, [2.5, 97.5])
            rows.append({"regime": regime, "p_wrong": p, "k": k, "gain_ci_low": lo, "gain_ci_high": hi})
    return pd.DataFrame(rows)


def bootstrap_b(raw, seeds, n_boot, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    rows = []
    for regime in REGIMES:
        n = len(raw[(regime, seeds[0])][0])
        names = list(raw[(regime, seeds[0])][1])
        samples = {name: [] for name in names + ["Δ AUROC (LR explanation − LR confidence)"]}
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            vals = {name: [] for name in samples}
            for s in seeds:
                wrong, scores = raw[(regime, s)]
                w = wrong[idx]
                if w.min() == w.max():
                    continue
                for name in names:
                    vals[name].append(roc_auc_score(w, scores[name][idx]))
                vals["Δ AUROC (LR explanation − LR confidence)"].append(
                    vals["LR: confidence + explanation"][-1] - vals["LR: confidence only"][-1])
            for name in samples:
                samples[name].append(np.mean(vals[name]))
        for name, vals in samples.items():
            lo, hi = np.percentile(vals, [2.5, 97.5])
            rows.append({"regime": regime, "score": name, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_a(df, noise_levels, out):
    agg = (df.groupby(["regime", "seed", "p_wrong", "k"]).balanced_accuracy.mean()
             .groupby(["regime", "p_wrong", "k"]).agg(["mean", "std"]).reset_index())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    styles = ["-", "--", "-.", ":"]
    for ax, regime in zip(axes, REGIMES):
        for p, ls in zip(noise_levels, styles):
            d = agg[(agg.regime == regime) & (agg.p_wrong == p)]
            ax.plot(d.k, d["mean"], ls, marker="o", ms=3, color=COLOURS[regime],
                    alpha=1.0 - 0.18 * noise_levels.index(p), label=f"clinician wrong {int(p * 100)}%")
            ax.fill_between(d.k, d["mean"] - d["std"], d["mean"] + d["std"],
                            color=COLOURS[regime], alpha=0.08)
        base = agg[(agg.regime == regime) & (agg.p_wrong == 0) & (agg.k == 0)]["mean"].item()
        ax.axhline(base, color="grey", lw=1, ls=":")
        ax.axvspan(0, 3, color="grey", alpha=0.07)
        ax.set_title(regime)
        ax.set_xlabel("concept groups corrected (most uncertain first)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("balanced accuracy (test)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Use case A: correction by a noisy clinician (shaded: ≤ 3 corrections; mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_b(df, ci, out):
    names = list(dict.fromkeys(df.score))
    agg = df.groupby(["regime", "score"]).auroc.agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    width = 0.27
    x = np.arange(len(names))
    for i, regime in enumerate(REGIMES):
        d = agg[agg.regime == regime].set_index("score").loc[names]
        c = ci[ci.regime == regime].set_index("score").loc[names]
        ax.bar(x + (i - 1) * width, d["mean"], width, color=COLOURS[regime], label=regime,
               yerr=[d["mean"] - c.ci_low, c.ci_high - d["mean"]], capsize=2)
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("AUROC for detecting a wrong diagnosis")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Use case B: do explanation-based signals flag errors? (mean over seeds, 95% bootstrap CI)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Optional: cases for a small human study with the group members
# ---------------------------------------------------------------------------

def export_study(run, out_dir, n=20, seed=0):
    """n test cases (half correct, half wrong) with diagnosis + predicted concepts only.
    study_cases.csv is what participants see; study_key.csv holds the answers."""
    rng = np.random.default_rng(seed)
    probs = f_probs(run, run["bottleneck"])
    pred = probs.argmax(1)
    ok = np.where(pred == run["y"])[0]
    bad = np.where(pred != run["y"])[0]
    pick = np.concatenate([rng.choice(ok, n // 2, replace=False), rng.choice(bad, n - n // 2, replace=False)])
    rng.shuffle(pick)
    labels = argmax_concepts(run["bottleneck"])
    cases, key = [], []
    for case_id, i in enumerate(pick, 1):
        row = {"case": case_id, "row_index": int(run["row_index"][i]),
               "model_diagnosis": DIAGNOSIS_CLASSES[pred[i]],
               "model_confidence": round(float(probs[i].max()), 2)}
        for g, grp in enumerate(CONCEPT_GROUPS):
            sl = SLICES[g]
            row[grp.abbrev] = f"{grp.classes[labels[i, g]]} ({run['bottleneck'][i, sl].max():.2f})"
        row["participant_answer (trust / dont_trust)"] = ""
        cases.append(row)
        key.append({"case": case_id, "true_diagnosis": DIAGNOSIS_CLASSES[run["y"][i]],
                    "model_correct": bool(pred[i] == run["y"][i])})
    pd.DataFrame(cases).to_csv(out_dir / "study_cases.csv", index=False)
    pd.DataFrame(key).to_csv(out_dir / "study_key.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=Path(__file__).resolve().parent / "cbm_derm7pt" / "results")
    parser.add_argument("--joint-lambda", default="1", help="selects results/final_lambda<L>_seed<S>")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    parser.add_argument("--n-draws", type=int, default=20, help="noise repetitions per run")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--export-study", action="store_true")
    args = parser.parse_args()

    res = Path(args.results_dir)
    out = res / f"part2_lambda{args.joint_lambda}"
    out.mkdir(parents=True, exist_ok=True)

    runs, val_runs = {}, {}
    for s in args.seeds:
        run_dir = res / f"final_lambda{args.joint_lambda}_seed{s}"
        for regime in REGIMES:
            runs[(regime, s)] = load_run(run_dir, regime, "test")
            val_runs[(regime, s)] = load_run(run_dir, regime, "valid")
    print(f"Loaded {len(args.seeds)} seeds x {len(REGIMES)} regimes (joint lambda = {args.joint_lambda})")

    # ---- Use case A
    df_a, preds = use_case_a(runs, args.noise, args.n_draws)
    df_a.to_csv(out / "use_case_a_all.csv", index=False)
    per_run = df_a.groupby(["regime", "seed", "p_wrong", "k"])[
        ["balanced_accuracy", "accuracy", "harm_rate", "fix_rate"]].mean().reset_index()
    base = per_run[per_run.k == 0].set_index(["regime", "seed", "p_wrong"]).balanced_accuracy
    per_run["gain"] = per_run.balanced_accuracy - per_run.set_index(["regime", "seed", "p_wrong"]).index.map(base)
    ci_a = bootstrap_a(runs, preds, args.seeds, args.n_boot, args.noise, args.n_draws)

    rows = []
    for regime in REGIMES:
        for p in args.noise:
            d = per_run[(per_run.regime == regime) & (per_run.p_wrong == p) & (per_run.k == 3)]
            c = ci_a[(ci_a.regime == regime) & (ci_a.p_wrong == p)].iloc[0]
            rows.append({"regime": regime, "clinician wrong": f"{int(p * 100)}%",
                         "bal. acc. k=0": mean_std(per_run[(per_run.regime == regime) & (per_run.p_wrong == p)
                                                           & (per_run.k == 0)].balanced_accuracy),
                         "bal. acc. k=3": mean_std(d.balanced_accuracy),
                         "gain k=3": mean_std(d.gain),
                         "gain 95% CI": f"[{c.gain_ci_low:+.3f}, {c.gain_ci_high:+.3f}]",
                         "fixed": mean_std(d.fix_rate), "harmed": mean_std(d.harm_rate),
                         "criterion met": "yes" if c.gain_ci_low > 0 else "no"})
    table_a = pd.DataFrame(rows)
    table_a.to_csv(out / "use_case_a_summary.csv", index=False)
    plot_a(df_a, args.noise, out / "use_case_a.png")

    # ---- Use case B
    df_b, raw = use_case_b(runs, val_runs)
    df_b.to_csv(out / "use_case_b_all.csv", index=False)
    ci_b = bootstrap_b(raw, args.seeds, args.n_boot)
    rows = []
    for regime in REGIMES:
        for name in dict.fromkeys(df_b.score):
            d = df_b[(df_b.regime == regime) & (df_b.score == name)]
            c = ci_b[(ci_b.regime == regime) & (ci_b.score == name)].iloc[0]
            rows.append({"regime": regime, "score": name, "AUROC": mean_std(d.auroc),
                         "95% CI": f"[{c.ci_low:.3f}, {c.ci_high:.3f}]"})
        c = ci_b[(ci_b.regime == regime) & (ci_b.score.str.startswith("Δ"))].iloc[0]
        rows.append({"regime": regime, "score": "Δ (explanation adds beyond confidence)",
                     "AUROC": "", "95% CI": f"[{c.ci_low:+.3f}, {c.ci_high:+.3f}]"})
    table_b = pd.DataFrame(rows)
    table_b.to_csv(out / "use_case_b_summary.csv", index=False)
    plot_b(df_b, ci_b[~ci_b.score.str.startswith("Δ")], out / "use_case_b.png")

    err = df_b.groupby("regime").error_rate.mean()

    with open(out / "summary.md", "w") as fh:
        fh.write(f"# Part 2 results (joint lambda = {args.joint_lambda}, seeds {args.seeds}, test split)\n\n")
        fh.write("## Use case A: correction of the 3 most uncertain concept groups\n\n")
        fh.write("Criterion met = the 95% bootstrap CI of the gain is above 0.\n\n")
        fh.write(table_a.to_markdown(index=False) + "\n\n")
        fh.write("## Use case B: AUROC for detecting wrong diagnoses\n\n")
        fh.write("Error rates: " + ", ".join(f"{r} {err[r]:.2f}" for r in REGIMES) + "\n\n")
        fh.write(table_b.to_markdown(index=False) + "\n")
    print((out / "summary.md").read_text())

    if args.export_study:
        export_study(runs[("independent", args.seeds[0])], out)
        print(f"Study cases written to {out}/study_cases.csv (answers in study_key.csv)")
    print(f"All outputs in {out}")


if __name__ == "__main__":
    main()
