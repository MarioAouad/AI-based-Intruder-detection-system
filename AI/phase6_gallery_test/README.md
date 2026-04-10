# Phase 6 — Multi-Template Gallery Test

> Tests 1-to-N identity matching by comparing live captures against a stored gallery of owner templates. Visualises the "Margin of Safety" between owners and intruders and validates the 0.65 cosine threshold.

---

## Purpose

Phase 5 proved that Facenet512 + cosine distance is the best model/metric combination. Phase 6 answers the critical deployment question: **"If we store 3 owner templates, can the system reliably unlock for the owner while blocking all intruders?"**

This is a **1-to-N matching** test — each probe image is compared against every gallery template, and the **minimum distance** decides the outcome.

---

## The 1-to-N Matching Logic

```
For each probe image:
    distances = [DeepFace.verify(probe, gallery[i]) for i in range(N)]
    min_distance = min(distances)
    
    if min_distance < THRESHOLD:
        → UNLOCK (predicted as Owner)
    else:
        → BLOCK (predicted as Intruder)
```

**Why min() instead of avg()?** A genuine owner may have one ideal template that closely matches the current angle/lighting. Averaging would dilute a strong match with weaker ones, increasing False Negatives.

---

## Threshold Selection: 0.65

| Threshold | Effect |
|-----------|--------|
| Too low (e.g., 0.40) | High Precision, but owners get locked out (low Recall) |
| Too high (e.g., 0.80) | All owners pass, but intruders breach too (low Precision) |
| **0.65** | **100% Precision + high Recall — the engineering sweet spot** |

The 0.65 threshold was selected because it maintains **zero False Positives** (no intruder has ever breached) while still achieving high Recall for owner captures. The Margin of Safety scatter plot visually confirms clear separation between owner distances (~0.3–0.5) and intruder distances (~0.8–1.0).

---

## Visual Dashboard (1×2 Layout)

### Left: Margin of Safety Scatter Plot

A 1D scatter/strip plot where:
- **Y-axis:** Cosine distance (0.0 to 1.2)
- **X-axis:** Two categories — "Owner Captures" (green dots) and "Intruders" (red dots)
- **Threshold line:** Bold dashed horizontal line at Y = 0.65
- **Outlier annotations:** Labels on the owner point closest to the line and the intruder point closest to breaching

**What to look for:** A wide gap between the top of the green cluster and the bottom of the red cluster. The wider this gap, the more robust the system is to edge cases.

### Right: Confusion Matrix + Metrics Card

- Colour-coded 2×2 table (TP/TN in green, FP/FN in red)
- Styled metrics card showing Accuracy, Precision, Recall, F1-Score

**Correct Matrix Layout:**

| | Predicted Owner | Predicted Intruder |
|---|---|---|
| **Actual Owner** | TP (green) | FN (red) |
| **Actual Intruder** | FP (red) | TN (green) |

---

## Folder Structure

```
phase6_gallery_test/
├── gallery_test.py            ← Gallery matching test script
├── gallery/                   ← Owner baseline templates (aligned 160×160)
├── live_captures/             ← Owner test shots (ground truth = match)
├── intruders/                 ← Non-owner faces (ground truth = no match)
└── results/                   ← Timestamped report PNGs
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
# 1. Place 1-3 owner templates in gallery/
# 2. Place owner test shots in live_captures/
# 3. Place intruder faces in intruders/
python phase6_gallery_test/gallery_test.py
```

---

## Configuration

All parameters are defined at the top of `gallery_test.py`:

```python
MODEL_NAME  = "Facenet512"
METRIC      = "cosine"
THRESHOLD   = 0.65
```

To experiment with different thresholds, edit `THRESHOLD` and re-run. The dashboard will reflect the new decision boundary automatically.
