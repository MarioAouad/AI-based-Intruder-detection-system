"""
custom_benchmark.py  –  Phase 5: Face Recognition Model Benchmarking
======================================================================

Evaluates 8 DeepFace recognition models on pre-aligned 160×160 face data
using Precision, Recall, F1-Score, and Confusion Matrix metrics.

Pairing Logic
-------------
  owners/    images  →  combinations(2)  →  Match pairs   (Ground Truth = True)
  intruders/ images  →  product(owners)  →  Mismatch pairs (Ground Truth = False)

Visual Dashboard
----------------
  2×2 matplotlib figure saved to results/ with timestamp.
    ┌──────────────────┬──────────────────────┐
    │ F1-Score (desc.)  │ Precision vs Recall  │
    ├──────────────────┼──────────────────────┤
    │ False Positives   │ Avg Inference Time   │
    └──────────────────┴──────────────────────┘

Dependencies
------------
    pip install deepface matplotlib numpy tf-keras

Usage
-----
    python phase5_benchmarking/custom_benchmark.py
"""

import os
import sys
import time
import itertools
import datetime

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR    = os.path.join(SCRIPT_DIR, "test_dataset")
OWNERS_DIR     = os.path.join(DATASET_DIR, "owners")
INTRUDERS_DIR  = os.path.join(DATASET_DIR, "intruders")
RESULTS_DIR    = os.path.join(SCRIPT_DIR, "results")

for d in [OWNERS_DIR, INTRUDERS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# DeepFace model list & cosine thresholds
# ---------------------------------------------------------------------------
METRIC = "cosine"

MODELS = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepID",
    "ArcFace",
    "SFace",
    "GhostFaceNet",
]

# Baseline cosine-distance thresholds per model.
# A pair is considered a "match" if distance < threshold.

THRESHOLDS = {
    "VGG-Face":      0.71,  
    "ArcFace":       0.50,  
    "SFace":         0.65,  
    "GhostFaceNet":  0.60,  
    "OpenFace":      0.40,  # Extreme drop to try and save it
    "DeepID":        0.01,  # Extreme drop
    "Facenet":       0.55,  
    "Facenet512":    0.65,  # Reverted to perfect baseline
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          HELPERS                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_image_paths(folder: str) -> list:
    """Return sorted list of absolute image paths in a folder."""
    if not os.path.isdir(folder):
        return []
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ])


def safe_divide(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 when denominator is zero."""
    return numerator / denominator if denominator > 0 else 0.0


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       PAIRING LOGIC                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_pairs(owner_imgs: list, intruder_imgs: list):
    """
    Generate all match and mismatch pairs.

    Match pairs   (ground_truth = True):
        Every combination of 2 owner images.
    Mismatch pairs (ground_truth = False):
        Every owner paired with every intruder.

    Returns list of (img_path_a, img_path_b, ground_truth_bool).
    """
    pairs = []

    # Matches: owner vs owner (every unique pair)
    for a, b in itertools.combinations(owner_imgs, 2):
        pairs.append((a, b, True))

    # Mismatches: owner vs intruder (full product)
    for a, b in itertools.product(owner_imgs, intruder_imgs):
        pairs.append((a, b, False))

    return pairs


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    EVALUATION ENGINE                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def evaluate_model(model_name: str, pairs: list) -> dict:
    """
    Run DeepFace.verify on every pair for a single model.
    Track TP, TN, FP, FN and average inference time.

    Returns a dict with keys:
        TP, TN, FP, FN, precision, recall, f1, avg_time, threshold
    """
    # Lazy-import so the startup banner only prints once,
    # and so we don't crash if deepface isn't installed yet.
    from deepface import DeepFace

    threshold = THRESHOLDS.get(model_name, 0.50)
    tp = tn = fp = fn = 0
    total_time = 0.0

    print(f"\n  Running {model_name} on {len(pairs)} pairs ...")

    for idx, (img_a, img_b, ground_truth) in enumerate(pairs):
        t0 = time.perf_counter()
        try:
            result = DeepFace.verify(
                img1_path=img_a,
                img2_path=img_b,
                model_name=model_name,
                distance_metric=METRIC,
                enforce_detection=False,   # data is already 160×160
            )
            distance = result.get("distance", 1.0)
            predicted_match = distance < threshold
        except Exception as e:
            print(f"    [WARN] Pair {idx+1} failed: {e}")
            predicted_match = False

        elapsed = time.perf_counter() - t0
        total_time += elapsed

        # ── Confusion matrix update ───────────────────────────────────
        if ground_truth:   # This pair SHOULD match (owner-owner)
            if predicted_match:
                tp += 1
            else:
                fn += 1
        else:              # This pair should NOT match (owner-intruder)
            if not predicted_match:
                tn += 1
            else:
                fp += 1

    # ── Derived metrics ───────────────────────────────────────────────
    precision = safe_divide(tp, tp + fp)
    recall    = safe_divide(tp, tp + fn)
    f1        = safe_divide(2.0 * precision * recall, precision + recall)
    accuracy  = safe_divide(tp + tn, tp + tn + fp + fn)
    avg_time  = safe_divide(total_time, len(pairs))

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "accuracy":  accuracy,
        "avg_time":  avg_time,
        "threshold": threshold,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    CONSOLE REPORT                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def print_report(model_name: str, metrics: dict) -> None:
    """Print a formatted console block for one model."""
    print(f"\n--- Model: {model_name} ---")
    print(f"Threshold Used        : {metrics['threshold']:.2f} (Cosine)")
    print(f"Confusion Matrix      : TP: {metrics['TP']} | TN: {metrics['TN']} "
          f"| FP: {metrics['FP']} | FN: {metrics['FN']}")
    print(f"Security (Precision)  : {metrics['precision'] * 100:.1f}%")
    print(f"Convenience (Recall)  : {metrics['recall'] * 100:.1f}%")
    print(f"F1-Score              : {metrics['f1']:.3f}")
    print(f"Overall Accuracy      : {metrics['accuracy'] * 100:.1f}%")
    print(f"Avg Match Time        : {metrics['avg_time']:.3f}s")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   VISUAL DASHBOARD                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def save_dashboard(all_results: dict) -> str:
    """
    Generate a 2×2 matplotlib dashboard and save to results/.

    Parameters
    ----------
    all_results : dict
        { model_name: metrics_dict }

    Returns
    -------
    str : path to saved figure
    """
    # ── Sort models by F1 descending for the F1 chart ─────────────────
    sorted_models = sorted(
        all_results.keys(),
        key=lambda m: all_results[m]["f1"],
        reverse=True,
    )

    # X-axis labels include the threshold  e.g. "SFace\n(0.63)"
    labels       = [f"{m}\n({all_results[m]['threshold']:.2f})" for m in sorted_models]
    f1_scores    = [all_results[m]["f1"]         for m in sorted_models]
    accuracies   = [all_results[m]["accuracy"]   for m in sorted_models]
    precisions   = [all_results[m]["precision"]  for m in sorted_models]
    recalls      = [all_results[m]["recall"]     for m in sorted_models]
    fps          = [all_results[m]["FP"]         for m in sorted_models]
    avg_times    = [all_results[m]["avg_time"]   for m in sorted_models]

    x = np.arange(len(sorted_models))

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Face Recognition Benchmark: Speed vs. Accuracy",
                 fontsize=16, fontweight="bold", y=0.98)

    # ── Top-Left: F1-Score vs Overall Accuracy (grouped bar) ──────────
    ax = axes[0, 0]
    width = 0.35
    bars_f1  = ax.bar(x - width / 2, f1_scores, width, color="#FFD700",
                      edgecolor="white", label="F1-Score")
    bars_acc = ax.bar(x + width / 2, accuracies, width, color="#DDA0DD",
                      edgecolor="white", label="Accuracy")
    ax.set_title("F1-Score vs Overall Accuracy", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score (0–1)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.25)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars_f1, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, val in zip(bars_acc, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ── Top-Right: Precision vs Recall (grouped) ─────────────────────
    ax = axes[0, 1]
    width = 0.35
    bars1 = ax.bar(x - width / 2, precisions, width, color="skyblue",
                   edgecolor="white", label="Precision")
    bars2 = ax.bar(x + width / 2, recalls, width, color="lightgreen",
                   edgecolor="white", label="Recall")
    ax.set_title("Precision vs Recall", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score (0–1)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.25)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars1, precisions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, val in zip(bars2, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ── Bottom-Left: False Positives / Security Breaches (salmon) ─────
    ax = axes[1, 0]
    bars = ax.bar(x, fps, color="salmon", edgecolor="white", linewidth=0.8)
    ax.set_title("False Positives — Security Breaches (Lower is Better)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    max_fp = max(fps) if fps and max(fps) > 0 else 1
    ax.set_ylim(0, max_fp * 1.35)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, fps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")

    # ── Bottom-Right: Average Inference Time (plum) ───────────────────
    ax = axes[1, 1]
    bars = ax.bar(x, avg_times, color="plum", edgecolor="white", linewidth=0.8)
    ax.set_title("Average Inference Time Per Pair", fontsize=12, fontweight="bold")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    max_t = max(avg_times) if avg_times and max(avg_times) > 0 else 1.0
    ax.set_ylim(0, max_t * 1.35)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, avg_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # ── Save with timestamp ───────────────────────────────────────────
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Benchmark_Report_{ts}.png"
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"\n[Dashboard] Saved → {filepath}")

    plt.show()
    plt.close(fig)
    return filepath


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                             MAIN                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print("=" * 60)
    print("  PHASE 5 — FACE RECOGNITION BENCHMARK")
    print("=" * 60)

    # ── Load images ───────────────────────────────────────────────────
    owner_imgs    = load_image_paths(OWNERS_DIR)
    intruder_imgs = load_image_paths(INTRUDERS_DIR)

    print(f"  Owner images    : {len(owner_imgs)}")
    print(f"  Intruder images : {len(intruder_imgs)}")

    if len(owner_imgs) < 2:
        print("\n[ERROR] Need at least 2 owner images for match-pair combinations.")
        print(f"  Place owner photos in: {OWNERS_DIR}")
        sys.exit(1)

    if len(intruder_imgs) < 1:
        print("\n[ERROR] Need at least 1 intruder image for mismatch pairs.")
        print(f"  Place intruder photos in: {INTRUDERS_DIR}")
        sys.exit(1)

    # ── Build pairs ───────────────────────────────────────────────────
    pairs = build_pairs(owner_imgs, intruder_imgs)
    n_match    = len(list(itertools.combinations(owner_imgs, 2)))
    n_mismatch = len(owner_imgs) * len(intruder_imgs)

    print(f"  Match pairs     : {n_match}   (owner-owner)")
    print(f"  Mismatch pairs  : {n_mismatch} (owner-intruder)")
    print(f"  Total pairs     : {len(pairs)}")
    print(f"  Models to test  : {len(MODELS)}")
    print("=" * 60)

    # ── Evaluate every model ──────────────────────────────────────────
    all_results = {}

    for model_name in MODELS:
        print(f"\n{'─' * 60}")
        print(f"  Benchmarking: {model_name}")
        print(f"{'─' * 60}")

        metrics = evaluate_model(model_name, pairs)
        all_results[model_name] = metrics
        print_report(model_name, metrics)

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  ALL MODELS COMPLETE — Generating dashboard...")
    print(f"{'=' * 60}")

    save_dashboard(all_results)
    print("\n[Benchmark] Done.")


if __name__ == "__main__":
    main()
