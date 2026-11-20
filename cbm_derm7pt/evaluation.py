"""Metrics for the diagnosis task, the concepts, and concept interventions."""

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)

from data import CONCEPT_GROUPS, DIAGNOSIS_CLASSES, group_slices, seven_point_score

SLICES = group_slices()
MEL = DIAGNOSIS_CLASSES.index("MEL")


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_cbm(model, X, device, intervention_mask=None, true_onehot=None):
    """Return (diagnosis probabilities (N, 5), bottleneck (N, 19)) as numpy arrays."""
    model.eval()
    x = torch.as_tensor(X, dtype=torch.float32, device=device)
    mask = None if intervention_mask is None else torch.as_tensor(intervention_mask, device=device)
    truth = None if true_onehot is None else torch.as_tensor(true_onehot, dtype=torch.float32, device=device)
    _, bottleneck, y_logits = model(x, mask, truth)
    return _probs(y_logits), bottleneck.cpu().numpy()


@torch.no_grad()
def predict_black_box(model, X, device):
    model.eval()
    x = torch.as_tensor(X, dtype=torch.float32, device=device)
    return _probs(model(x))


def _probs(logits):
    p = torch.softmax(logits.double(), 1).cpu().numpy()
    return p / p.sum(1, keepdims=True)


def bottleneck_to_labels(bottleneck):
    """(N, 19) concept probabilities -> (N, 7) argmax label per group."""
    return np.stack([bottleneck[:, sl].argmax(1) for sl in SLICES], axis=1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def task_metrics(y_true, probs):
    y_pred = probs.argmax(1)
    labels = list(range(len(DIAGNOSIS_CLASSES)))
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "melanoma_sensitivity": float(np.mean(y_pred[y_true == MEL] == MEL)),
        "melanoma_specificity": float(np.mean(y_pred[y_true != MEL] != MEL)),
    }
    try:
        metrics["macro_auc_ovr"] = roc_auc_score(y_true, probs, multi_class="ovr", labels=labels)
    except ValueError:  # a class missing from y_true
        metrics["macro_auc_ovr"] = float("nan")
    metrics["melanoma_auc"] = roc_auc_score(y_true == MEL, probs[:, MEL])
    for name, r in zip(DIAGNOSIS_CLASSES, per_class_recall):
        metrics[f"recall_{name}"] = r
    return {k: float(v) for k, v in metrics.items()}


def task_confusion(y_true, probs):
    return confusion_matrix(y_true, probs.argmax(1), labels=list(range(len(DIAGNOSIS_CLASSES))))


def concept_metrics(c_true, bottleneck):
    """Accuracy, balanced accuracy and macro F1 for each concept group, plus their means."""
    c_pred = bottleneck_to_labels(bottleneck)
    rows = []
    for g, group in enumerate(CONCEPT_GROUPS):
        labels = list(range(len(group.classes)))
        rows.append({
            "concept": group.abbrev,
            "accuracy": accuracy_score(c_true[:, g], c_pred[:, g]),
            "balanced_accuracy": balanced_accuracy_score(c_true[:, g], c_pred[:, g]),
            "macro_f1": f1_score(c_true[:, g], c_pred[:, g], labels=labels, average="macro", zero_division=0),
            "majority_class_accuracy": np.bincount(c_true[:, g]).max() / len(c_true),
        })
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "concept"}
    return rows, mean


def seven_point_rule_metrics(c_labels, y_true, threshold=3):
    """Melanoma detection with the clinical 7-point rule (score >= threshold -> melanoma)."""
    score = seven_point_score(c_labels)
    pred_mel = score >= threshold
    is_mel = y_true == MEL
    return {
        "melanoma_sensitivity": float(np.mean(pred_mel[is_mel])),
        "melanoma_specificity": float(np.mean(~pred_mel[~is_mel])),
        "melanoma_auc": float(roc_auc_score(is_mel, score)),
    }


# ---------------------------------------------------------------------------
# Interventions (Koh et al. 2020, Section 6)
# ---------------------------------------------------------------------------

def group_entropy(bottleneck):
    """Normalised entropy of each predicted concept group, (N, 7), in [0, 1]."""
    ent = []
    for sl in SLICES:
        p = np.clip(bottleneck[:, sl], 1e-12, 1.0)
        ent.append(-(p * np.log(p)).sum(1) / np.log(sl.stop - sl.start))
    return np.stack(ent, axis=1)


def single_group_interventions(model, X, c_onehot, y_true, device):
    """Change in accuracy / balanced accuracy when ONE concept group is set to the truth."""
    probs, _ = predict_cbm(model, X, device)
    base_acc = accuracy_score(y_true, probs.argmax(1))
    base_bacc = balanced_accuracy_score(y_true, probs.argmax(1))
    rows = []
    for g, group in enumerate(CONCEPT_GROUPS):
        mask = np.zeros((len(X), len(CONCEPT_GROUPS)), dtype=bool)
        mask[:, g] = True
        p, _ = predict_cbm(model, X, device, mask, c_onehot)
        rows.append({"concept": group.abbrev,
                     "delta_accuracy": accuracy_score(y_true, p.argmax(1)) - base_acc,
                     "delta_balanced_accuracy": balanced_accuracy_score(y_true, p.argmax(1)) - base_bacc})
    return rows


def intervention_curve(model, X, c_true, c_onehot, y_true, device, n_random_orders=20, seed=0):
    """Diagnosis performance when the first k concept groups (k = 0..7) are set to the truth.

    Two orderings are evaluated:
      * 'random'    - a random group order, averaged over `n_random_orders` permutations
                      (the same order for all patients, as in Koh et al.);
      * 'uncertain' - per patient, the groups whose predicted distribution has the highest
                      entropy are corrected first (what a clinician would check first).
    """
    rng = np.random.default_rng(seed)
    n, n_groups = c_true.shape
    _, bottleneck = predict_cbm(model, X, device)
    entropy_order = np.argsort(-group_entropy(bottleneck), axis=1)

    rows = []
    for k in range(n_groups + 1):
        # Random orders
        accs, baccs = [], []
        for _ in range(n_random_orders):
            mask = np.zeros((n, n_groups), dtype=bool)
            mask[:, rng.permutation(n_groups)[:k]] = True
            probs, _ = predict_cbm(model, X, device, mask, c_onehot)
            accs.append(accuracy_score(y_true, probs.argmax(1)))
            baccs.append(balanced_accuracy_score(y_true, probs.argmax(1)))
        rows.append({"k": k, "order": "random", "accuracy": np.mean(accs), "accuracy_std": np.std(accs),
                     "balanced_accuracy": np.mean(baccs), "balanced_accuracy_std": np.std(baccs)})

        # Most-uncertain-first order
        mask = np.zeros((n, n_groups), dtype=bool)
        np.put_along_axis(mask, entropy_order[:, :k], True, axis=1)
        probs, _ = predict_cbm(model, X, device, mask, c_onehot)
        rows.append({"k": k, "order": "uncertain",
                     "accuracy": accuracy_score(y_true, probs.argmax(1)), "accuracy_std": 0.0,
                     "balanced_accuracy": balanced_accuracy_score(y_true, probs.argmax(1)),
                     "balanced_accuracy_std": 0.0})
    return rows
