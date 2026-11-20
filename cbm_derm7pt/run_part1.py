"""Part 1: train the CBM with the independent, sequential and joint regimes and evaluate them.

Usage (after extract_features.py):
    python run_part1.py --data-dir ../release_v0 --features features/resnet50_derm.npz

Outputs are written to results/<run-name>/ (see README.md for the list of files).
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import (CONCEPT_GROUPS, CONCEPT_UNIT_NAMES, DIAGNOSIS_CLASSES, concepts_to_onehot,
                  load_derm7pt)
from evaluation import (bottleneck_to_labels, concept_metrics, intervention_curve,
                        predict_black_box, predict_cbm, seven_point_rule_metrics,
                        single_group_interventions, task_confusion, task_metrics)
from training import (TrainConfig, Weights, train_black_box, train_concept_predictor,
                      train_independent, train_joint, train_sequential)

REGIMES = ["independent", "sequential", "joint"]


def load_features(paths, n_rows):
    """Load one or more .npz feature files and concatenate them along the feature axis."""
    arrays = []
    for p in paths:
        npz = np.load(p)
        assert npz["X"].shape[0] == n_rows, f"{p} has {npz['X'].shape[0]} rows, meta.csv has {n_rows}"
        assert np.array_equal(npz["row_index"], np.arange(n_rows)), f"{p} is not in meta.csv row order"
        arrays.append(npz["X"])
    return np.concatenate(arrays, axis=1)


def standardise(X, train_idx):
    """Z-score features using statistics of the TRAIN split only."""
    mean, std = X[train_idx].mean(0), X[train_idx].std(0) + 1e-6
    return (X - mean) / std


def prediction_table(split_name, idx, y, c, probs, bottleneck=None):
    df = pd.DataFrame({"row_index": idx, "split": split_name, "y_true": [DIAGNOSIS_CLASSES[i] for i in y[idx]]})
    for k, name in enumerate(DIAGNOSIS_CLASSES):
        df[f"p_{name}"] = probs[:, k]
    df["y_pred"] = [DIAGNOSIS_CLASSES[i] for i in probs.argmax(1)]
    for g, group in enumerate(CONCEPT_GROUPS):
        df[f"true_{group.abbrev}"] = [group.classes[i] for i in c[idx, g]]
    if bottleneck is not None:
        c_pred = bottleneck_to_labels(bottleneck)
        for g, group in enumerate(CONCEPT_GROUPS):
            df[f"pred_{group.abbrev}"] = [group.classes[i] for i in c_pred[:, g]]
        for j, unit in enumerate(CONCEPT_UNIT_NAMES):
            df[f"c_{unit}"] = bottleneck[:, j]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to the unzipped release_v0 folder")
    parser.add_argument("--features", nargs="+", required=True, help="One or more .npz feature files")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--joint-lambda", type=float, default=1.0)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--n-random-orders", type=int, default=20)
    parser.add_argument("--no-plots", action="store_true")
    # The black-box baseline can use its own (validation-selected) hyper-parameters.
    parser.add_argument("--bb-lr", type=float, default=None)
    parser.add_argument("--bb-weight-decay", type=float, default=None)
    parser.add_argument("--bb-hidden-dim", type=int, default=None)
    parser.add_argument("--bb-dropout", type=float, default=None)
    args = parser.parse_args()

    cfg = TrainConfig(lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim,
                      dropout=args.dropout, joint_lambda=args.joint_lambda,
                      class_weights=not args.no_class_weights,
                      device="cuda" if torch.cuda.is_available() else "cpu", verbose=True)

    run_name = args.run_name or "_".join(Path(p).stem for p in args.features) + f"_seed{args.seed}"
    out = Path("results") / run_name
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- Data ----------------
    data = load_derm7pt(args.data_dir)
    print(data.summary())
    split = data.splits
    X = standardise(load_features(args.features, len(data.meta)), split["train"])
    y, C = data.y, data.c
    C_onehot = concepts_to_onehot(C)
    weights = Weights(y[split["train"]], C[split["train"]], cfg)

    # ---------------- Training ----------------
    seed = args.seed
    print("Training concept predictor g (shared by independent and sequential)")
    g, g_info = train_concept_predictor(X, C, split, cfg, weights, seed)
    models, train_info = {}, {"g": g_info}
    print("Training independent f")
    models["independent"], train_info["independent"] = train_independent(g, X, C_onehot, y, split, cfg, weights, seed)
    print("Training sequential f")
    models["sequential"], train_info["sequential"] = train_sequential(g, X, y, split, cfg, weights, seed)
    print("Training joint CBM")
    models["joint"], train_info["joint"] = train_joint(X, C, y, split, cfg, weights, seed)
    print("Training black-box baseline")
    cfg_bb = replace(cfg, **{k: v for k, v in {"lr": args.bb_lr, "weight_decay": args.bb_weight_decay,
                                               "hidden_dim": args.bb_hidden_dim, "dropout": args.bb_dropout}.items()
                             if v is not None})
    black_box, train_info["black_box"] = train_black_box(X, y, split, cfg_bb, weights, seed)

    # ---------------- Evaluation ----------------
    task_rows, concept_rows, rule_rows, interv_rows, single_rows, pred_tables = [], [], [], [], [], []
    confusions = {}

    for split_name in ("valid", "test"):
        idx = split[split_name]
        Xs, ys, Cs, Cs_onehot = X[idx], y[idx], C[idx], C_onehot[idx]

        # 7-point checklist rule on ground-truth concepts: the clinical reference point.
        rule_rows.append({"split": split_name, "concepts": "ground_truth", **seven_point_rule_metrics(Cs, ys)})

        for regime in REGIMES:
            model = models[regime]
            probs, bottleneck = predict_cbm(model, Xs, cfg.device)
            task_rows.append({"split": split_name, "model": regime, **task_metrics(ys, probs)})
            confusions[f"{regime}_{split_name}"] = task_confusion(ys, probs).tolist()

            per_group, mean = concept_metrics(Cs, bottleneck)
            concept_rows += [{"split": split_name, "model": regime, **r} for r in per_group]
            concept_rows.append({"split": split_name, "model": regime, "concept": "MEAN", **mean})

            rule_rows.append({"split": split_name, "concepts": f"predicted_{regime}",
                              **seven_point_rule_metrics(bottleneck_to_labels(bottleneck), ys)})

            # f applied to the TRUE concepts = the model after intervening on all 7 groups.
            full_mask = np.ones_like(Cs, dtype=bool)
            oracle_probs, _ = predict_cbm(model, Xs, cfg.device, full_mask, Cs_onehot)
            task_rows.append({"split": split_name, "model": f"{regime}_true_concepts",
                              **task_metrics(ys, oracle_probs)})

            interv_rows += [{"split": split_name, "model": regime, **r} for r in
                            intervention_curve(model, Xs, Cs, Cs_onehot, ys, cfg.device,
                                               args.n_random_orders, seed)]
            single_rows += [{"split": split_name, "model": regime, **r} for r in
                            single_group_interventions(model, Xs, Cs_onehot, ys, cfg.device)]
            pred_tables.append(prediction_table(split_name, idx, y, C, probs, bottleneck).assign(model=regime))

        bb_probs = predict_black_box(black_box, Xs, cfg.device)
        task_rows.append({"split": split_name, "model": "black_box", **task_metrics(ys, bb_probs)})
        confusions[f"black_box_{split_name}"] = task_confusion(ys, bb_probs).tolist()
        pred_tables.append(prediction_table(split_name, idx, y, C, bb_probs).assign(model="black_box"))

    # ---------------- Save ----------------
    pd.DataFrame(task_rows).to_csv(out / "task_metrics.csv", index=False)
    pd.DataFrame(concept_rows).to_csv(out / "concept_metrics.csv", index=False)
    pd.DataFrame(rule_rows).to_csv(out / "seven_point_rule.csv", index=False)
    pd.DataFrame(interv_rows).to_csv(out / "interventions.csv", index=False)
    pd.DataFrame(single_rows).to_csv(out / "single_concept_interventions.csv", index=False)
    pd.concat(pred_tables).to_csv(out / "predictions.csv", index=False)

    # Weights of the linear label predictor f: the global explanation of each CBM.
    weight_rows = []
    for regime in REGIMES:
        W = models[regime].f.linear.weight.detach().cpu().numpy()
        b = models[regime].f.linear.bias.detach().cpu().numpy()
        for k, cls in enumerate(DIAGNOSIS_CLASSES):
            weight_rows.append({"model": regime, "class": cls, "bias": b[k],
                                **dict(zip(CONCEPT_UNIT_NAMES, W[k]))})
    pd.DataFrame(weight_rows).to_csv(out / "label_predictor_weights.csv", index=False)

    with open(out / "confusion_matrices.json", "w") as fh:
        json.dump({"classes": DIAGNOSIS_CLASSES, **confusions}, fh, indent=2)
    with open(out / "run_config.json", "w") as fh:
        json.dump({"args": vars(args), "train_config": asdict(cfg), "black_box_config": asdict(cfg_bb), "training": train_info,
                   "split_sizes": {k: int(len(v)) for k, v in split.items()}}, fh, indent=2)
    torch.save({r: m.state_dict() for r, m in models.items()} | {"black_box": black_box.state_dict()},
               out / "models.pt")

    # ---------------- Summary ----------------
    summary = pd.DataFrame(task_rows)
    summary = summary[summary.split == "test"][["model", "accuracy", "balanced_accuracy", "macro_f1",
                                                 "macro_auc_ovr", "melanoma_sensitivity"]]
    print("\nTest set - diagnosis\n" + summary.round(3).to_string(index=False))
    cm = pd.DataFrame(concept_rows)
    cm = cm[(cm.split == "test") & (cm.concept == "MEAN")][["model", "accuracy", "balanced_accuracy", "macro_f1"]]
    print("\nTest set - mean concept performance\n" + cm.round(3).to_string(index=False))

    if not args.no_plots:
        from plots import make_all_plots
        make_all_plots(out)
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
