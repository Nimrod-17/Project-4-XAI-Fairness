"""Loading and label encoding for the derm7pt dataset.

The label grouping replicates `Derm7PtDatasetGroupInfrequent` from the official
derm7pt repository (https://github.com/jeremykawahara/derm7pt), re-implemented in
plain pandas so that the project does not depend on keras / old numpy versions.

Diagnosis (5 classes): BCC, NEV, MEL, MISC, SK
Concepts (7-point checklist, grouped as in the official repo):
    PN  pigment network        : absent / typical / atypical
    BWV blue whitish veil      : absent / present
    VS  vascular structures    : absent / regular / irregular
    PIG pigmentation           : absent / regular / irregular
    STR streaks                : absent / regular / irregular
    DaG dots and globules      : absent / regular / irregular
    RS  regression structures  : absent / present
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Label definitions
# ---------------------------------------------------------------------------

DIAGNOSIS_CLASSES = ["BCC", "NEV", "MEL", "MISC", "SK"]

DIAGNOSIS_MAP = {
    "basal cell carcinoma": "BCC",
    "nevus": "NEV",
    "blue nevus": "NEV",
    "clark nevus": "NEV",
    "combined nevus": "NEV",
    "congenital nevus": "NEV",
    "dermal nevus": "NEV",
    "recurrent nevus": "NEV",
    "reed or spitz nevus": "NEV",
    "melanoma": "MEL",
    "melanoma (in situ)": "MEL",
    "melanoma (less than 0.76 mm)": "MEL",
    "melanoma (0.76 to 1.5 mm)": "MEL",
    "melanoma (more than 1.5 mm)": "MEL",
    "melanoma metastasis": "MEL",
    "dermatofibroma": "MISC",
    "lentigo": "MISC",
    "melanosis": "MISC",
    "miscellaneous": "MISC",
    "vascular lesion": "MISC",
    "seborrheic keratosis": "SK",
}


@dataclass(frozen=True)
class ConceptGroup:
    """One criterion of the 7-point checklist, encoded as a multi-class concept."""

    abbrev: str
    column: str  # column name in meta.csv
    classes: tuple  # class names, index = numeric label
    raw_to_class: dict  # raw string in meta.csv -> class name
    checklist_class: str  # the class that adds points to the 7-point score
    checklist_points: int  # 2 = major criterion, 1 = minor criterion


CONCEPT_GROUPS = [
    ConceptGroup(
        "PN", "pigment_network", ("absent", "typical", "atypical"),
        {"absent": "absent", "typical": "typical", "atypical": "atypical"},
        "atypical", 2,
    ),
    ConceptGroup(
        "BWV", "blue_whitish_veil", ("absent", "present"),
        {"absent": "absent", "present": "present"},
        "present", 2,
    ),
    ConceptGroup(
        "VS", "vascular_structures", ("absent", "regular", "irregular"),
        {
            "absent": "absent",
            "arborizing": "regular", "comma": "regular", "hairpin": "regular",
            "within regression": "regular", "wreath": "regular",
            "dotted": "irregular", "linear irregular": "irregular",
        },
        "irregular", 2,
    ),
    ConceptGroup(
        "PIG", "pigmentation", ("absent", "regular", "irregular"),
        {
            "absent": "absent",
            "diffuse regular": "regular", "localized regular": "regular",
            "diffuse irregular": "irregular", "localized irregular": "irregular",
        },
        "irregular", 1,
    ),
    ConceptGroup(
        "STR", "streaks", ("absent", "regular", "irregular"),
        {"absent": "absent", "regular": "regular", "irregular": "irregular"},
        "irregular", 1,
    ),
    ConceptGroup(
        "DaG", "dots_and_globules", ("absent", "regular", "irregular"),
        {"absent": "absent", "regular": "regular", "irregular": "irregular"},
        "irregular", 1,
    ),
    ConceptGroup(
        "RS", "regression_structures", ("absent", "present"),
        {
            "absent": "absent",
            "blue areas": "present", "white areas": "present", "combinations": "present",
        },
        "present", 1,
    ),
]

GROUP_SIZES = [len(g.classes) for g in CONCEPT_GROUPS]  # [3, 2, 3, 3, 3, 3, 2]
N_CONCEPT_UNITS = sum(GROUP_SIZES)  # 19 units in the bottleneck
CONCEPT_UNIT_NAMES = [f"{g.abbrev}:{c}" for g in CONCEPT_GROUPS for c in g.classes]


def group_slices():
    """Return the slice of the bottleneck vector that belongs to each concept group."""
    slices, start = [], 0
    for size in GROUP_SIZES:
        slices.append(slice(start, start + size))
        start += size
    return slices


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _normalise(series):
    return series.astype(str).str.strip().str.lower()


def encode_diagnosis(series):
    raw = _normalise(series)
    unknown = sorted(set(raw) - set(DIAGNOSIS_MAP))
    if unknown:
        raise ValueError(f"Unknown diagnosis strings in meta.csv: {unknown}")
    return raw.map(DIAGNOSIS_MAP).map(DIAGNOSIS_CLASSES.index).to_numpy(dtype=np.int64)


def encode_concept(series, group):
    raw = _normalise(series)
    unknown = sorted(set(raw) - set(group.raw_to_class))
    if unknown:
        raise ValueError(f"Unknown values for {group.column}: {unknown}")
    return raw.map(group.raw_to_class).map(list(group.classes).index).to_numpy(dtype=np.int64)


def concepts_to_onehot(concept_labels):
    """(N, 7) integer concept labels -> (N, 19) one-hot bottleneck vector."""
    n = concept_labels.shape[0]
    onehot = np.zeros((n, N_CONCEPT_UNITS), dtype=np.float32)
    for g, sl in enumerate(group_slices()):
        onehot[np.arange(n), sl.start + concept_labels[:, g]] = 1.0
    return onehot


def seven_point_score(concept_labels):
    """7-point checklist score from (N, 7) integer concept labels (score >= 3 suggests melanoma)."""
    score = np.zeros(concept_labels.shape[0], dtype=np.int64)
    for g, group in enumerate(CONCEPT_GROUPS):
        positive = group.classes.index(group.checklist_class)
        score += group.checklist_points * (concept_labels[:, g] == positive)
    return score


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class Derm7pt:
    meta: pd.DataFrame  # full meta.csv (row order = feature order)
    y: np.ndarray  # (N,) diagnosis labels
    c: np.ndarray  # (N, 7) concept labels
    splits: dict  # {"train": idx, "valid": idx, "test": idx}
    image_dir: Path

    def image_paths(self, image_type="derm"):
        if image_type not in ("derm", "clinic"):
            raise ValueError("image_type must be 'derm' or 'clinic'")
        return [self.image_dir / p for p in self.meta[image_type]]

    def summary(self):
        lines = []
        for split, idx in self.splits.items():
            counts = np.bincount(self.y[idx], minlength=len(DIAGNOSIS_CLASSES))
            dist = ", ".join(f"{k}={v}" for k, v in zip(DIAGNOSIS_CLASSES, counts))
            lines.append(f"{split:>5}: n={len(idx):4d} | {dist}")
        return "\n".join(lines)


def load_derm7pt(release_dir):
    """Load meta.csv and the official train/valid/test split of the derm7pt release.

    `release_dir` is the unzipped `release_v0` folder, containing `images/` and `meta/`.
    """
    release_dir = Path(release_dir)
    meta = pd.read_csv(release_dir / "meta" / "meta.csv")

    y = encode_diagnosis(meta["diagnosis"])
    c = np.stack([encode_concept(meta[g.column], g) for g in CONCEPT_GROUPS], axis=1)

    splits = {}
    for split in ("train", "valid", "test"):
        idx = pd.read_csv(release_dir / "meta" / f"{split}_indexes.csv")["indexes"].to_numpy()
        splits[split] = idx.astype(np.int64)

    # Sanity checks: the three splits must be disjoint and cover every case exactly once.
    all_idx = np.concatenate(list(splits.values()))
    assert len(np.unique(all_idx)) == len(all_idx), "Overlapping indexes between splits"
    assert np.array_equal(np.sort(all_idx), np.arange(len(meta))), "Splits do not cover all cases"

    return Derm7pt(meta=meta, y=y, c=c, splits=splits, image_dir=release_dir / "images")
