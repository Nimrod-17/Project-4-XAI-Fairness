# Concept bottleneck models for derm7pt — Part 1

Concept bottleneck model (CBM, Koh et al. 2020) that predicts 5 diagnoses
(BCC, NEV, MEL, MISC, SK) through the 7-point checklist criteria, trained with the
**independent**, **sequential** and **joint** regimes.

## Setup

```bash
pip install -r requirements.txt
```

Download the data from http://derm.cs.sfu.ca and unzip it so that the folder looks like:

```
Prj 4/
├── derm7pt/          # official repo (only used as reference, not imported)
├── release_v0/
│   ├── images/
│   └── meta/         # meta.csv, train_indexes.csv, valid_indexes.csv, test_indexes.csv
└── cbm_derm7pt/      # this code
```

## Run

```bash
cd cbm_derm7pt
# 1. Embeddings from a frozen ImageNet backbone (once per image type / backbone)
python extract_features.py --data-dir ../release_v0 --image-type derm --backbone resnet50
# 2. Train and evaluate the three regimes + a black-box baseline
python run_part1.py --data-dir ../release_v0 --features features/resnet50_derm.npz
```

Useful options: `--image-type clinic`, several feature files at once
(`--features features/resnet50_derm.npz features/resnet50_clinic.npz` concatenates them),
`--joint-lambda`, `--hidden-dim 0` (linear g), `--seed`, `--no-class-weights`.

## Design

| Component | Choice | Why |
|---|---|---|
| Data split | Official `train/valid/test_indexes.csv` (413 / 203 / 395) | Comparable with the literature; checked to be disjoint and complete |
| Labels | Same grouping as `Derm7PtDatasetGroupInfrequent` | 5 diagnoses, 7 concept groups with 2–3 classes each |
| x | Embedding of a frozen ImageNet backbone (ResNet-50 by default), z-scored with train statistics | Small dataset; every run takes seconds, so all regimes and seeds are cheap |
| g: x → c | MLP (1 hidden layer) with one softmax per concept group → 19-unit bottleneck | Each group is a probability distribution over its classes |
| f: c → y | Single linear layer | Diagnosis depends on the image **only** through the concepts; weights are readable |
| Losses | Cross-entropy for y; mean of 7 cross-entropies for c; balanced class weights | Strong class imbalance (NEV dominates, SK and BCC are rare) |
| Model selection | Early stopping on the validation loss | Test set is used only for the final evaluation |

**Regimes**

* *Independent*: g trained on x → c; f trained on the ground-truth one-hot concepts.
  At test time f receives g's predicted probabilities.
* *Sequential*: the same g; f trained on g's predicted probabilities (g frozen).
* *Joint*: g and f trained together on `L_y + lambda * L_c`.

**Interventions**: the predicted distribution of k concept groups is replaced by the
ground-truth one-hot vector (k = 0…7). Two orders: random (averaged over 20 permutations)
and "most uncertain first" (highest predicted entropy per patient).

**Baselines**: a black-box MLP x → y with the same capacity as g, and the clinical
7-point checklist rule (score ≥ 3 → melanoma) on ground-truth and predicted concepts.

## Outputs (`results/<run-name>/`)

| File | Content |
|---|---|
| `task_metrics.csv` | Accuracy, balanced accuracy, macro F1, macro AUC, melanoma sensitivity/specificity, per-class recall. `<regime>_true_concepts` = f applied to the true concepts |
| `concept_metrics.csv` | Per-concept accuracy, balanced accuracy, macro F1 and majority-class baseline |
| `seven_point_rule.csv` | 7-point rule on true vs predicted concepts |
| `interventions.csv`, `single_concept_interventions.csv` | Intervention curves and the effect of correcting each concept alone |
| `label_predictor_weights.csv` | Weights of f (global explanation of each CBM) |
| `predictions.csv` | Per-patient probabilities, predicted and true concepts (also needed for Part 3) |
| `confusion_matrices.json`, `run_config.json`, `models.pt` | |
| `*.png` | Interventions, f weights, concept performance, task performance, confusion matrices |
