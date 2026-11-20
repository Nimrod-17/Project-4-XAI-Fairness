"""The three CBM training regimes of Koh et al. (2020), plus a black-box baseline.

* Independent: g is trained on x -> c. f is trained on the TRUE concepts c -> y.
               At test time f receives g's predictions: y_hat = f(g(x)).
* Sequential:  g is trained on x -> c (the same g as above). f is then trained on g's
               PREDICTED concepts g(x) -> y, with g frozen.
* Joint:       g and f are trained together, minimising L_y + lambda * L_c.

Every model is selected by early stopping on the validation split; the test split is
never used for training or model selection.
"""

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data import CONCEPT_GROUPS, DIAGNOSIS_CLASSES
from models import (BlackBox, ConceptBottleneckModel, ConceptPredictor, LabelPredictor,
                    concept_loss, group_softmax)


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 500
    patience: int = 40  # epochs without validation improvement before stopping
    hidden_dim: int = 256  # hidden units of g (0 = linear g)
    dropout: float = 0.3
    joint_lambda: float = 1.0  # weight of the concept loss in joint training
    class_weights: bool = True  # balanced class weights for diagnosis and concept losses
    device: str = "cpu"
    verbose: bool = False


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def balanced_weights(labels, n_classes):
    """sklearn-style 'balanced' class weights: n_samples / (n_classes * count)."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    weights = len(labels) / (n_classes * np.maximum(counts, 1.0))
    weights[counts == 0] = 0.0
    return torch.tensor(weights, dtype=torch.float32)


class Weights:
    """Class weights for the diagnosis loss and for each concept-group loss (train split only)."""

    def __init__(self, y_train, c_train, cfg):
        if cfg.class_weights:
            self.y = balanced_weights(y_train, len(DIAGNOSIS_CLASSES)).to(cfg.device)
            self.c = [balanced_weights(c_train[:, g], len(grp.classes)).to(cfg.device)
                      for g, grp in enumerate(CONCEPT_GROUPS)]
        else:
            self.y, self.c = None, None


def fit(model, loss_fn, train_tensors, val_tensors, cfg, seed):
    """Generic mini-batch training with early stopping on the validation loss.

    loss_fn(model, *batch) -> scalar loss. The weights from the best validation epoch are
    restored at the end.
    """
    model.to(cfg.device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(*train_tensors), batch_size=cfg.batch_size,
                        shuffle=True, generator=generator)
    val_tensors = [t.to(cfg.device) for t in val_tensors]

    best_loss, best_state, best_epoch, waited = float("inf"), None, 0, 0
    for epoch in range(cfg.max_epochs):
        model.train()
        for batch in loader:
            batch = [t.to(cfg.device) for t in batch]
            optimiser.zero_grad()
            loss = loss_fn(model, *batch)
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model, *val_tensors).item()
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, best_epoch, waited = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            waited += 1
            if waited >= cfg.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    if cfg.verbose:
        print(f"    best epoch {best_epoch}, val loss {best_loss:.4f}")
    return model, {"best_epoch": best_epoch, "best_val_loss": best_loss}


def _t(a, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def train_concept_predictor(X, C, split, cfg, weights, seed):
    """Train g: x -> c on the train split (shared by the independent and sequential CBMs)."""
    set_seed(seed)
    g = ConceptPredictor(X.shape[1], cfg.hidden_dim, cfg.dropout)

    def loss_fn(model, x, c):
        return concept_loss(model(x), c, weights.c)

    tr, va = split["train"], split["valid"]
    return fit(g, loss_fn, [_t(X[tr]), _t(C[tr], torch.long)],
               [_t(X[va]), _t(C[va], torch.long)], cfg, seed)


def train_label_predictor(B_train, y_train, B_val, y_val, cfg, weights, seed):
    """Train f: bottleneck -> y on fixed bottleneck inputs (true or predicted concepts)."""
    set_seed(seed)
    f = LabelPredictor(len(DIAGNOSIS_CLASSES))

    def loss_fn(model, b, y):
        return F.cross_entropy(model(b), y, weight=weights.y)

    return fit(f, loss_fn, [_t(B_train), _t(y_train, torch.long)],
               [_t(B_val), _t(y_val, torch.long)], cfg, seed)


@torch.no_grad()
def predicted_bottleneck(g, X, device):
    g.eval()
    return group_softmax(g(_t(X).to(device))).cpu().numpy()


# ---------------------------------------------------------------------------
# The three regimes
# ---------------------------------------------------------------------------

def train_independent(g, X, C_onehot, y, split, cfg, weights, seed):
    tr, va = split["train"], split["valid"]
    # f only ever sees ground-truth concepts during training (and validation).
    f, info = train_label_predictor(C_onehot[tr], y[tr], C_onehot[va], y[va], cfg, weights, seed)
    return ConceptBottleneckModel(copy.deepcopy(g), f).eval(), info


def train_sequential(g, X, y, split, cfg, weights, seed):
    tr, va = split["train"], split["valid"]
    # f is trained on g's predicted concept probabilities; g stays frozen.
    B = predicted_bottleneck(g, X, cfg.device)
    f, info = train_label_predictor(B[tr], y[tr], B[va], y[va], cfg, weights, seed)
    return ConceptBottleneckModel(copy.deepcopy(g), f).eval(), info


def train_joint(X, C, y, split, cfg, weights, seed):
    set_seed(seed)
    model = ConceptBottleneckModel(
        ConceptPredictor(X.shape[1], cfg.hidden_dim, cfg.dropout),
        LabelPredictor(len(DIAGNOSIS_CLASSES)),
    )

    def loss_fn(m, x, c, yy):
        concept_logits, _, y_logits = m(x)
        task = F.cross_entropy(y_logits, yy, weight=weights.y)
        return task + cfg.joint_lambda * concept_loss(concept_logits, c, weights.c)

    tr, va = split["train"], split["valid"]
    return fit(model, loss_fn,
               [_t(X[tr]), _t(C[tr], torch.long), _t(y[tr], torch.long)],
               [_t(X[va]), _t(C[va], torch.long), _t(y[va], torch.long)], cfg, seed)


def train_black_box(X, y, split, cfg, weights, seed):
    set_seed(seed)
    model = BlackBox(X.shape[1], len(DIAGNOSIS_CLASSES), cfg.hidden_dim, cfg.dropout)

    def loss_fn(m, x, yy):
        return F.cross_entropy(m(x), yy, weight=weights.y)

    tr, va = split["train"], split["valid"]
    return fit(model, loss_fn, [_t(X[tr]), _t(y[tr], torch.long)],
               [_t(X[va]), _t(y[va], torch.long)], cfg, seed)
