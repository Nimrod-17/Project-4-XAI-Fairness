"""Concept bottleneck model (Koh et al., 2020) for derm7pt.

    x (image embedding) --g--> c (7 concept groups, 19 units) --f--> y (5 diagnoses)

* g, the concept predictor, outputs one logit vector per concept group. The bottleneck is
  the concatenation of the per-group softmax probabilities, so each group is a probability
  distribution over its classes (e.g. PN: absent / typical / atypical).
* f, the label predictor, is a single linear layer on the 19 bottleneck units. The diagnosis
  therefore depends on the image *only* through the predicted concepts, and each weight
  f.W[k, j] reads directly as "how much concept unit j pushes towards diagnosis k".
* Interventions replace the predicted distribution of selected groups with the ground-truth
  one-hot vector, as in Koh et al. Section 6.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import GROUP_SIZES, N_CONCEPT_UNITS, group_slices

SLICES = group_slices()


def mlp(in_dim, out_dim, hidden_dim, dropout):
    """A linear layer if hidden_dim == 0, else a one-hidden-layer MLP."""
    if hidden_dim == 0:
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, out_dim))
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


def group_softmax(concept_logits):
    """Apply a softmax within each concept group: (N, 19) logits -> (N, 19) probabilities."""
    return torch.cat([F.softmax(concept_logits[:, sl], dim=1) for sl in SLICES], dim=1)


def concept_loss(concept_logits, concept_labels, group_weights=None):
    """Mean cross-entropy over the 7 concept groups.

    concept_labels: (N, 7) integer labels. group_weights: optional list with one class-weight
    tensor per group (to counter class imbalance within each concept).
    """
    losses = []
    for g, sl in enumerate(SLICES):
        w = None if group_weights is None else group_weights[g]
        losses.append(F.cross_entropy(concept_logits[:, sl], concept_labels[:, g], weight=w))
    return torch.stack(losses).mean()


class ConceptPredictor(nn.Module):
    """g: image embedding -> concept logits."""

    def __init__(self, in_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = mlp(in_dim, N_CONCEPT_UNITS, hidden_dim, dropout)

    def forward(self, x):
        return self.net(x)


class LabelPredictor(nn.Module):
    """f: bottleneck (19 concept probabilities) -> diagnosis logits. Linear on purpose."""

    def __init__(self, n_classes=5):
        super().__init__()
        self.linear = nn.Linear(N_CONCEPT_UNITS, n_classes)

    def forward(self, c):
        return self.linear(c)


class ConceptBottleneckModel(nn.Module):
    def __init__(self, concept_predictor, label_predictor):
        super().__init__()
        self.g = concept_predictor
        self.f = label_predictor

    def forward(self, x, intervention_mask=None, true_concepts_onehot=None):
        """
        intervention_mask: optional (N, 7) bool tensor; True = replace that group with the truth.
        true_concepts_onehot: (N, 19) one-hot ground truth, required when intervening.
        Returns concept logits, bottleneck (after interventions) and diagnosis logits.
        """
        concept_logits = self.g(x)
        bottleneck = group_softmax(concept_logits)
        if intervention_mask is not None:
            unit_mask = expand_group_mask(intervention_mask)
            bottleneck = torch.where(unit_mask, true_concepts_onehot, bottleneck)
        return concept_logits, bottleneck, self.f(bottleneck)


def expand_group_mask(group_mask):
    """(N, 7) group mask -> (N, 19) unit mask."""
    sizes = torch.tensor(GROUP_SIZES, device=group_mask.device)
    return torch.repeat_interleave(group_mask.long(), sizes, dim=1).bool()


class BlackBox(nn.Module):
    """Standard (non-interpretable) baseline: image embedding -> diagnosis, same capacity as g."""

    def __init__(self, in_dim, n_classes=5, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = mlp(in_dim, n_classes, hidden_dim, dropout)

    def forward(self, x):
        return self.net(x)
