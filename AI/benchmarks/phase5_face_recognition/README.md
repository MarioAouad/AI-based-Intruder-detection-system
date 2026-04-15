# Phase 5 — Face Recognition Model Benchmarking

> Evaluates 8 DeepFace recognition models using Precision, Recall, F1-Score, and Overall Accuracy to select the best model and metric for identity verification.

---

## Purpose

After the preprocessing pipeline produces standardised 160×160 faces, we need to choose which **recognition model** can most reliably distinguish the owner from intruders. This benchmark tests 8 models with the cosine distance metric and compares them across security (Precision) and convenience (Recall) dimensions.

---

## Models Tested

| Model | Cosine Threshold | Architecture |
|-------|-----------------|--------------|
| VGG-Face | 0.75 | VGG-16 fine-tuned on faces |
| Facenet | 0.60 | Google's Inception-ResNet v1 |
| **Facenet512** | **0.72** | **512-d embedding variant (selected)** |
| OpenFace | 0.55 | Torch-based, lightweight |
| DeepID | 0.15 | Early deep face model |
| ArcFace | 0.68 | Additive Angular Margin Loss |
| SFace | 0.72 | Sigmoid-constrained hypersphere |
| GhostFaceNet | 0.70 | GhostNet backbone, efficient |

*Thresholds were tuned from baseline defaults to reduce False Negatives while maintaining low False Positives.*

---

## Pairing Logic

**Match pairs** (Ground Truth = True):
`itertools.combinations(owner_images, 2)` — every unique pair of owner images.

**Mismatch pairs** (Ground Truth = False):
`itertools.product(owner_images, intruder_images)` — every owner vs. every intruder.

---

## Confusion Matrix Mapping

| | Predicted Match | Predicted Non-Match |
|---|---|---|
| **Actual Match** (owner-owner) | TP | FN |
| **Actual Non-Match** (owner-intruder) | FP | TN |

**Derived Metrics:**
- **Precision** = TP / (TP + FP) — *"Of all unlocks, how many were correct?"*
- **Recall** = TP / (TP + FN) — *"Of all owners, how many were let in?"*
- **F1-Score** = 2 × (P × R) / (P + R) — harmonic mean
- **Accuracy** = (TP + TN) / total

---

## Visual Dashboard (2×2 Grid)

| Panel | Chart Type | Colour |
|-------|-----------|--------|
| Top-Left | F1-Score vs Overall Accuracy | Gold + Plum (grouped bar) |
| Top-Right | Precision vs Recall | Sky Blue + Light Green |
| Bottom-Left | False Positives (Security Breaches) | Salmon |
| Bottom-Right | Average Inference Time per Pair | Plum |

All charts include data labels and model-specific thresholds on the X-axis.

---

## Winning Selection

**Facenet512** was selected for the production pipeline:

- **Embedding Dimensionality:** 512-d vectors provide richer discriminative features than 128-d Facenet.
- **F1-Score:** Consistently top-2 across threshold sweeps.
- **Cosine Metric:** Chosen over Euclidean — normalised similarity is more robust to embedding magnitude variations.
- **Speed:** Moderate inference time, acceptable for batch gallery matching.

---

## Folder Structure

```
phase5_benchmarking/
├── custom_benchmark.py        ← Benchmarking script
├── test_dataset/
│   ├── owners/                ← Owner face images (≥2 required)
│   └── intruders/             ← Intruder face images (≥1 required)
└── results/                   ← Timestamped dashboard PNGs
```

---

## Requirements

```
deepface>=0.0.80
tf-keras>=2.16.0
matplotlib>=3.7.0
numpy>=1.24.0
```

---

## Usage

```bash
# Place images in test_dataset/owners/ and test_dataset/intruders/
python phase5_benchmarking/custom_benchmark.py
```
