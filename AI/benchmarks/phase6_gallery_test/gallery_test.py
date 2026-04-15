"""
gallery_test.py  –  Phase 6: Multi-Template Gallery Matching Test Suite
========================================================================

Tests 1-to-N identity matching: each probe image (live capture or intruder)
is compared against ALL gallery templates.  The minimum cosine distance
decides whether the probe is accepted (UNLOCK) or rejected (BLOCKED).

Folder Layout
-------------
  phase6_gallery_test/
  ├── gallery/          ← Owner baseline templates (≥1 aligned 160×160)
  ├── live_captures/    ← Owner test shots (ground truth = match)
  ├── intruders/        ← Non-owner faces   (ground truth = no match)
  ├── results/          ← Saved report PNGs
  └── gallery_test.py   ← This script

Visual Dashboard (1×2)
----------------------
  ┌────────────────────────┬──────────────────────────┐
  │ Margin-of-Safety       │ Confusion Matrix &       │
  │ Scatter Plot           │ Final Metrics            │
  └────────────────────────┴──────────────────────────┘

Dependencies
------------
    pip install deepface matplotlib numpy tf-keras

Usage
-----
    python phase6_gallery_test/gallery_test.py
"""

import os
import sys
import time
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
GALLERY_DIR     = os.path.join(SCRIPT_DIR, "gallery")
LIVE_DIR        = os.path.join(SCRIPT_DIR, "live_captures")
INTRUDERS_DIR   = os.path.join(SCRIPT_DIR, "intruders")
RESULTS_DIR     = os.path.join(SCRIPT_DIR, "results")

for d in [GALLERY_DIR, LIVE_DIR, INTRUDERS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# DeepFace configuration
# ---------------------------------------------------------------------------
MODEL_NAME  = "Facenet512"
METRIC      = "cosine"
THRESHOLD   = 0.65

IMAGE_EXTS  = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


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


def safe_divide(num: float, den: float) -> float:
    """Division returning 0.0 when denominator is zero."""
    return num / den if den > 0 else 0.0


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   1-to-N MATCHING ENGINE                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def match_against_gallery(probe_path: str, gallery_paths: list) -> float:
    """
    Compare a single probe image against every gallery template.
    Returns the MINIMUM cosine distance across all comparisons.

    A lower min-distance means a closer match to the owner.
    """
    from deepface import DeepFace

    distances = []
    for gal_path in gallery_paths:
        try:
            result = DeepFace.verify(
                img1_path=probe_path,
                img2_path=gal_path,
                model_name=MODEL_NAME,
                distance_metric=METRIC,
                enforce_detection=False,
            )
            distances.append(result.get("distance", 1.0))
        except Exception as e:
            print(f"    [WARN] Pair failed ({os.path.basename(gal_path)}): {e}")
            distances.append(1.0)

    return min(distances) if distances else 1.0


def run_evaluation(gallery_paths: list, live_paths: list, intruder_paths: list):
    """
    Evaluate all live captures and intruders against the gallery.

    Returns
    -------
    dict with keys:
        owner_results   : list of (filename, min_dist, passed_bool)
        intruder_results: list of (filename, min_dist, passed_bool)
        TP, TN, FP, FN  : int counts
        precision, recall, f1, accuracy : float metrics
    """
    tp = tn = fp = fn = 0

    owner_results    = []   # (filename, min_dist, passed)
    intruder_results = []

    # ── Owner live captures (ground truth = SHOULD match) ─────────────
    print(f"\n{'─' * 60}")
    print(f"  Testing {len(live_paths)} OWNER live captures")
    print(f"{'─' * 60}")

    for img_path in live_paths:
        name = os.path.basename(img_path)
        min_dist = match_against_gallery(img_path, gallery_paths)
        is_match = min_dist < THRESHOLD

        if is_match:
            tp += 1
            tag = "UNLOCK (Pass)"
        else:
            fn += 1
            tag = "LOCKED (Fail)"

        owner_results.append((name, min_dist, is_match))
        print(f"[LIVE CAPTURE] {name} | Min Distance: {min_dist:.4f} | Result: {tag}")

    # ── Intruder probes (ground truth = should NOT match) ─────────────
    print(f"\n{'─' * 60}")
    print(f"  Testing {len(intruder_paths)} INTRUDER probes")
    print(f"{'─' * 60}")

    for img_path in intruder_paths:
        name = os.path.basename(img_path)
        min_dist = match_against_gallery(img_path, gallery_paths)
        is_match = min_dist < THRESHOLD

        if not is_match:
            tn += 1
            tag = "BLOCKED (Pass)"
        else:
            fp += 1
            tag = "BREACH! (Fail)"

        intruder_results.append((name, min_dist, not is_match))
        print(f"[INTRUDER] {name} | Min Distance: {min_dist:.4f} | Result: {tag}")

    # ── Derived metrics ───────────────────────────────────────────────
    precision = safe_divide(tp, tp + fp)
    recall    = safe_divide(tp, tp + fn)
    f1        = safe_divide(2.0 * precision * recall, precision + recall)
    accuracy  = safe_divide(tp + tn, tp + tn + fp + fn)

    return {
        "owner_results":    owner_results,
        "intruder_results": intruder_results,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "accuracy":  accuracy,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    CONSOLE REPORT                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def print_final_report(metrics: dict) -> None:
    """Print a formatted summary block."""
    print(f"\n{'=' * 60}")
    print("  FINAL METRICS")
    print(f"{'=' * 60}")
    print(f"  Model             : {MODEL_NAME}")
    print(f"  Metric            : {METRIC}")
    print(f"  Threshold         : {THRESHOLD}")
    print(f"  Confusion Matrix  : TP={metrics['TP']}  TN={metrics['TN']}  "
          f"FP={metrics['FP']}  FN={metrics['FN']}")
    print(f"  Overall Accuracy  : {metrics['accuracy'] * 100:.1f}%")
    print(f"  Precision         : {metrics['precision'] * 100:.1f}%")
    print(f"  Recall            : {metrics['recall'] * 100:.1f}%")
    print(f"  F1-Score          : {metrics['f1']:.3f}")
    print(f"{'=' * 60}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   VISUAL DASHBOARD                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def save_dashboard(metrics: dict) -> str:
    """
    Generate a 1×2 dashboard:
        Left  – Margin-of-Safety scatter plot
        Right – Confusion matrix table + metrics card

    Returns path to the saved PNG.
    """
    owner_results    = metrics["owner_results"]
    intruder_results = metrics["intruder_results"]

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    fig, (ax_scatter, ax_table) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Phase 6: Multi-Template Margin of Safety – Test Run: {now_str}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ──────────────────────────────────────────────────────────────────
    # LEFT SUBPLOT: Margin-of-Safety Scatter Plot
    # ──────────────────────────────────────────────────────────────────
    np.random.seed(42)

    owner_dists = [d for _, d, _ in owner_results]
    owner_x     = np.zeros(len(owner_dists)) + np.random.uniform(-0.12, 0.12, len(owner_dists))

    intruder_dists = [d for _, d, _ in intruder_results]
    intruder_x     = np.ones(len(intruder_dists)) + np.random.uniform(-0.12, 0.12, len(intruder_dists))

    ax_scatter.scatter(
        owner_x, owner_dists,
        color="#2ecc71", edgecolors="white", s=90, alpha=0.85, zorder=5,
        label="Owner Captures",
    )
    ax_scatter.scatter(
        intruder_x, intruder_dists,
        color="#e74c3c", edgecolors="white", s=90, alpha=0.85, zorder=5,
        label="Intruders",
    )

    # Decision boundary
    ax_scatter.axhline(
        y=THRESHOLD, color="#2c3e50", linewidth=2.5,
        linestyle="--", zorder=4,
    )
    ax_scatter.text(
        1.45, THRESHOLD, f"Security Threshold ({THRESHOLD})",
        va="center", ha="left", fontsize=9, fontweight="bold",
        color="#2c3e50",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#2c3e50", alpha=0.9),
    )

    # Outlier: owner closest to crossing the line (highest distance)
    if owner_dists:
        worst_owner_idx = int(np.argmax(owner_dists))
        worst_owner_val = owner_dists[worst_owner_idx]
        ax_scatter.annotate(
            f"{worst_owner_val:.3f}",
            xy=(owner_x[worst_owner_idx], worst_owner_val),
            xytext=(owner_x[worst_owner_idx] + 0.20, worst_owner_val + 0.04),
            fontsize=8, fontweight="bold", color="#27ae60",
            arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2),
        )

    # Outlier: intruder closest to breaching (lowest distance)
    if intruder_dists:
        worst_intruder_idx = int(np.argmin(intruder_dists))
        worst_intruder_val = intruder_dists[worst_intruder_idx]
        ax_scatter.annotate(
            f"{worst_intruder_val:.3f}",
            xy=(intruder_x[worst_intruder_idx], worst_intruder_val),
            xytext=(intruder_x[worst_intruder_idx] + 0.20, worst_intruder_val - 0.04),
            fontsize=8, fontweight="bold", color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
        )

    ax_scatter.set_xticks([0, 1])
    ax_scatter.set_xticklabels(["Owner\nCaptures", "Intruders"], fontsize=10, fontweight="bold")
    ax_scatter.set_ylabel("Cosine Distance", fontsize=11)
    ax_scatter.set_ylim(0.0, 1.2)
    ax_scatter.set_xlim(-0.5, 1.8)
    ax_scatter.legend(loc="upper left", fontsize=9)
    ax_scatter.grid(axis="y", linestyle="--", alpha=0.35)
    ax_scatter.set_title("Margin of Safety", fontsize=12, fontweight="bold")

    # ──────────────────────────────────────────────────────────────────
    # RIGHT SUBPLOT: Confusion Matrix table + Metrics Card
    # ──────────────────────────────────────────────────────────────────
    ax_table.axis("off")

    tp, tn = metrics["TP"], metrics["TN"]
    fp, fn = metrics["FP"], metrics["FN"]

    # Rows = Actual, Columns = Predicted
    # Row 1 (Actual Owner):    TP = correctly unlocked | FN = wrongly locked
    # Row 2 (Actual Intruder): FP = wrongly unlocked   | TN = correctly blocked
    cell_text = [
        [str(tp), str(fn)],
        [str(fp), str(tn)],
    ]
    row_labels = ["Actual\nOwner", "Actual\nIntruder"]
    col_labels = ["Predicted\nOwner", "Predicted\nIntruder"]

    cell_colours = [
        ["#a8e6cf", "#ffb3b3"],   # TP green,  FN red
        ["#ffb3b3", "#a8e6cf"],   # FP red,    TN green
    ]

    tbl = ax_table.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colours,
        cellLoc="center",
        bbox=[0.25, 0.55, 0.65, 0.35],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(14)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#555555")
        cell.set_linewidth(1.2)
        if row >= 1 and col >= 0:
            cell.set_text_props(fontweight="bold", fontsize=18)
        else:
            cell.set_text_props(fontweight="bold", fontsize=10)

    ax_table.set_title("Confusion Matrix", fontsize=12, fontweight="bold", pad=15)

    # Metrics Card
    """card_text = (
        f"  Overall Accuracy :  {metrics['accuracy'] * 100:.1f}%\n"
        f"  Precision        :  {metrics['precision'] * 100:.1f}%\n"
        f"  Recall           :  {metrics['recall'] * 100:.1f}%\n"
        f"  F1-Score         :  {metrics['f1']:.3f}"
    )"""
    card_text = (
        f"Overall Accuracy : {metrics['accuracy'] * 100:>5.1f}%\n"
        f"Precision        : {metrics['precision'] * 100:>5.1f}%\n"
        f"Recall           : {metrics['recall'] * 100:>5.1f}%\n"
        f"F1-Score         : {metrics['f1']:>6.3f} "
    )
    ax_table.text(
        0.57, 0.38, card_text,
        transform=ax_table.transAxes,
        ha="center", va="top",
        fontsize=11, fontweight="bold", fontfamily="monospace",
        bbox=dict(
            boxstyle="square,pad=0.6",
            facecolor="#f5f5f5",
            edgecolor="#aaaaaa",
            linewidth=1.5,
        ),
    )

    # ──────────────────────────────────────────────────────────────────
    # Save & show
    # ──────────────────────────────────────────────────────────────────
    plt.tight_layout()

    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Phase6_Report_{ts}.png"
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"\n[Dashboard] Saved \u2192 {filepath}")

    plt.show()
    plt.close(fig)
    return filepath


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                             MAIN                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print("=" * 60)
    print("  PHASE 6 — MULTI-TEMPLATE GALLERY TEST")
    print("=" * 60)

    gallery_imgs  = load_image_paths(GALLERY_DIR)
    live_imgs     = load_image_paths(LIVE_DIR)
    intruder_imgs = load_image_paths(INTRUDERS_DIR)

    print(f"  Gallery templates   : {len(gallery_imgs)}")
    print(f"  Live captures       : {len(live_imgs)}")
    print(f"  Intruder probes     : {len(intruder_imgs)}")
    print(f"  Model               : {MODEL_NAME}")
    print(f"  Metric / Threshold  : {METRIC} / {THRESHOLD}")
    print("=" * 60)

    if len(gallery_imgs) == 0:
        print(f"\n[ERROR] No gallery templates found.")
        print(f"  Place owner baseline images in: {GALLERY_DIR}")
        sys.exit(1)

    if len(live_imgs) == 0 and len(intruder_imgs) == 0:
        print(f"\n[ERROR] No test images found.")
        print(f"  Place owner test shots in : {LIVE_DIR}")
        print(f"  Place intruder faces in   : {INTRUDERS_DIR}")
        sys.exit(1)

    # ── Run evaluation ────────────────────────────────────────────────
    metrics = run_evaluation(gallery_imgs, live_imgs, intruder_imgs)

    # ── Console report ────────────────────────────────────────────────
    print_final_report(metrics)

    # ── Visual dashboard ──────────────────────────────────────────────
    print("\n  Generating dashboard...")
    save_dashboard(metrics)
    print("\n[Gallery Test] Done.")


if __name__ == "__main__":
    main()
