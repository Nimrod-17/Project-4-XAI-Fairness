"""Figures for Part 1, generated from the CSV/JSON files in a results folder.

Usage: python plots.py results/<run-name>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import CONCEPT_GROUPS, CONCEPT_UNIT_NAMES

REGIMES = ["independent", "sequential", "joint"]
COLOURS = {"independent": "#2a78b5", "sequential": "#e08a1e", "joint": "#3a9a5b", "black_box": "#7a7a7a"}


def plot_interventions(out, split="test"):
    df = pd.read_csv(out / "interventions.csv")
    df = df[df.split == split]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, metric in zip(axes, ["accuracy", "balanced_accuracy"]):
        for regime in REGIMES:
            for order, style in [("random", "-"), ("uncertain", "--")]:
                d = df[(df.model == regime) & (df.order == order)].sort_values("k")
                ax.plot(d.k, d[metric], style, marker="o", ms=3, color=COLOURS[regime],
                        label=f"{regime} ({order})")
                if order == "random":
                    ax.fill_between(d.k, d[metric] - d[f"{metric}_std"], d[metric] + d[f"{metric}_std"],
                                    color=COLOURS[regime], alpha=0.12)
        ax.set_xlabel("Number of concept groups intervened on")
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Concept interventions ({split} set)")
    fig.tight_layout()
    fig.savefig(out / f"interventions_{split}.png", dpi=200)
    plt.close(fig)


def plot_label_weights(out):
    df = pd.read_csv(out / "label_predictor_weights.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    vmax = np.abs(df[CONCEPT_UNIT_NAMES].to_numpy()).max()
    for ax, regime in zip(axes, REGIMES):
        d = df[df.model == regime]
        im = ax.imshow(d[CONCEPT_UNIT_NAMES].to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_yticks(range(len(d)), d["class"])
        ax.set_xticks(range(len(CONCEPT_UNIT_NAMES)), CONCEPT_UNIT_NAMES, rotation=90, fontsize=7)
        ax.set_title(regime)
        # Separators between concept groups
        pos = 0
        for group in CONCEPT_GROUPS[:-1]:
            pos += len(group.classes)
            ax.axvline(pos - 0.5, color="k", lw=0.6)
    fig.colorbar(im, ax=axes, shrink=0.8, label="weight of f")
    fig.suptitle("Label predictor weights (concept unit -> diagnosis logit)")
    fig.savefig(out / "label_predictor_weights.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_concept_performance(out, split="test", metric="balanced_accuracy"):
    df = pd.read_csv(out / "concept_metrics.csv")
    df = df[df.split == split]
    concepts = [g.abbrev for g in CONCEPT_GROUPS] + ["MEAN"]
    width = 0.27
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, regime in enumerate(REGIMES):
        d = df[df.model == regime].set_index("concept").loc[concepts]
        ax.bar(np.arange(len(concepts)) + (i - 1) * width, d[metric], width, color=COLOURS[regime], label=regime)
    ax.set_xticks(range(len(concepts)), concepts)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(0, 1)
    ax.axhline(1 / 3, color="k", ls=":", lw=0.8)
    ax.legend()
    ax.set_title(f"Concept prediction ({split} set)")
    fig.tight_layout()
    fig.savefig(out / f"concept_{metric}_{split}.png", dpi=200)
    plt.close(fig)


def plot_task_performance(out, split="test"):
    df = pd.read_csv(out / "task_metrics.csv")
    df = df[(df.split == split) & df.model.isin(REGIMES + ["black_box"])]
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc_ovr", "melanoma_sensitivity"]
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, model in enumerate(REGIMES + ["black_box"]):
        d = df[df.model == model].iloc[0]
        ax.bar(np.arange(len(metrics)) + (i - 1.5) * width, [d[m] for m in metrics], width,
               color=COLOURS[model], label=model)
    ax.set_xticks(range(len(metrics)), [m.replace("_", " ") for m in metrics])
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, fontsize=8)
    ax.set_title(f"Diagnosis performance ({split} set)")
    fig.tight_layout()
    fig.savefig(out / f"task_metrics_{split}.png", dpi=200)
    plt.close(fig)


def plot_confusions(out, split="test"):
    with open(out / "confusion_matrices.json") as fh:
        cms = json.load(fh)
    classes = cms["classes"]
    models = REGIMES + ["black_box"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, model in zip(axes, models):
        cm = np.array(cms[f"{model}_{split}"])
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(classes)), classes)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("predicted")
        ax.set_title(model)
    axes[0].set_ylabel("true")
    fig.tight_layout()
    fig.savefig(out / f"confusion_{split}.png", dpi=200)
    plt.close(fig)


def make_all_plots(out):
    out = Path(out)
    plot_interventions(out)
    plot_label_weights(out)
    plot_concept_performance(out)
    plot_task_performance(out)
    plot_confusions(out)


if __name__ == "__main__":
    make_all_plots(sys.argv[1])
