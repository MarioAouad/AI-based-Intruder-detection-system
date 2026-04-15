"""
benchmark_faces.py  –  Phase 3: Face Detection Model Benchmark
================================================================

Tests 4 face detection engines on cropped torso images saved by main_watchdog.py
and produces:

  1. A 2×2 visual grid per image (saved to benchmark_results/) showing each
     model's bounding box — ideal for a university report.
  2. A formatted console table summarising avg speed (FPS) and detection
     success rate across all test images.

The 4 Models
-------------
  ┌────┬─────────────────────────────────────────────────────────────────┐
  │  1 │ OpenCV Haar Cascade  – classic Viola-Jones, CPU, no DL needed  │
  │  2 │ OpenCV YuNet (DNN)   – lightweight ONNX CNN, very fast         │
  │  3 │ MTCNN                – multi-task cascaded CNN, high accuracy   │
  │  4 │ MediaPipe Face Det.  – Google's GPU-friendly BlazeFace model   │
  └────┴─────────────────────────────────────────────────────────────────┘

Environment Setup
-----------------
  pip install opencv-python opencv-contrib-python mtcnn mediapipe

YuNet weights (ONNX)
--------------------
  Download face_detection_yunet_2023mar.onnx from:
  https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

  Place it in the same directory as this script, or update YUNET_MODEL_PATH below.

Usage
-----
  python benchmark_faces.py
  python benchmark_faces.py --input path/to/images --output path/to/results
"""

import os
import sys
import time
import argparse
import glob
import datetime
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DIR = os.path.join(SCRIPT_DIR, "captured_targets")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "benchmark_results")

# ── YuNet ONNX weight file ────────────────────────────────────────────────
# Download from:
#   https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
# Place alongside this script or change the path below.
YUNET_MODEL_PATH = os.path.join(SCRIPT_DIR, "face_detection_yunet_2023mar.onnx")

# Haar cascade XML shipped with every OpenCV install.
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Model display names and colours (BGR) for the 2×2 grid.
MODEL_NAMES  = ["Haar Cascade", "YuNet (DNN)", "MTCNN", "MediaPipe"]
MODEL_COLORS = [
    (255, 100,  50),   # Haar   – blue-ish
    (  0, 220, 220),   # YuNet  – yellow
    (  0, 200,   0),   # MTCNN  – green
    (200,  50, 255),   # MP     – magenta
]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                  PER-MODEL INFERENCE FUNCTIONS                         ║
# ║  Each function takes a BGR image and returns:                          ║
# ║      (faces: list[tuple(x,y,w,h)],  inference_ms: float)              ║
# ╚══════════════════════════════════════════════════════════════════════════╝


# ── 1. Haar Cascade ──────────────────────────────────────────────────────

def detect_haar(image: np.ndarray, cascade: cv2.CascadeClassifier) -> Tuple[List, float]:
    """
    Classic Viola-Jones detector.
    Converts to greyscale internally; returns faces as (x, y, w, h) rects.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)          # Improve contrast for Haar

    t0 = time.perf_counter()
    rects = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    elapsed_ms = (time.perf_counter() - t0) * 1_000

    faces = [(x, y, w, h) for (x, y, w, h) in rects]
    return faces, elapsed_ms


# ── 2. YuNet (OpenCV DNN) ────────────────────────────────────────────────

def detect_yunet(
    image: np.ndarray,
    detector: Optional[cv2.FaceDetectorYN],
) -> Tuple[List, float]:
    """
    YuNet is a lightweight CNN-based face detector shipped via OpenCV Zoo.
    It outputs bounding boxes, landmarks, and confidence scores.
    """
    if detector is None:
        return [], 0.0

    h, w = image.shape[:2]
    detector.setInputSize((w, h))

    t0 = time.perf_counter()
    _, raw_detections = detector.detect(image)
    elapsed_ms = (time.perf_counter() - t0) * 1_000

    faces = []
    if raw_detections is not None:
        for det in raw_detections:
            # YuNet returns [x, y, w, h, ...landmarks..., score]
            x, y, bw, bh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            faces.append((x, y, bw, bh))

    return faces, elapsed_ms


# ── 3. MTCNN ──────────────────────────────────────────────────────────────

def detect_mtcnn(image: np.ndarray, mtcnn_detector) -> Tuple[List, float]:
    """
    Multi-task Cascaded Convolutional Network.
    Expects RGB input; we convert from BGR inside this function.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    t0 = time.perf_counter()
    results = mtcnn_detector.detect_faces(rgb)
    elapsed_ms = (time.perf_counter() - t0) * 1_000

    faces = []
    for r in results:
        x, y, w, h = r["box"]
        # MTCNN can return negative coords; clamp them.
        x, y = max(0, x), max(0, y)
        faces.append((x, y, w, h))

    return faces, elapsed_ms


# ── 4. MediaPipe Face Detection ──────────────────────────────────────────

def detect_mediapipe(
    image: np.ndarray,
    mp_face_detection,
) -> Tuple[List, float]:
    """
    Google MediaPipe's BlazeFace SSD model.
    Expects RGB input.  Returns normalised bounding boxes which we
    convert to pixel coordinates.
    """
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    t0 = time.perf_counter()
    results = mp_face_detection.process(rgb)
    elapsed_ms = (time.perf_counter() - t0) * 1_000

    faces = []
    if results.detections:
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            # Convert normalised [0-1] coords to pixel coords.
            x  = int(bb.xmin * w)
            y  = int(bb.ymin * h)
            bw = int(bb.width * w)
            bh = int(bb.height * h)
            x, y = max(0, x), max(0, y)
            faces.append((x, y, bw, bh))

    return faces, elapsed_ms


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                           GRID BUILDER                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_grid(
    image: np.ndarray,
    all_faces: List[List],
    model_names: List[str],
    model_colors: List[Tuple],
    times_ms: List[float],
) -> np.ndarray:
    """
    Create a 2×2 panel image.  Each cell shows the original image with
    one model's bounding boxes drawn, plus a label with the model name,
    number of faces found, and inference time.
    """
    panels = []

    for i in range(4):
        panel = image.copy()
        colour = model_colors[i]
        faces  = all_faces[i]

        # Draw every face box for this model.
        for (x, y, w, h) in faces:
            cv2.rectangle(panel, (x, y), (x + w, y + h), colour, 2)

        # Label at top-left
        label = f"{model_names[i]}  |  {len(faces)} face(s)  |  {times_ms[i]:.1f} ms"
        cv2.putText(
            panel, label, (6, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            panel, label, (6, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, colour, 1, cv2.LINE_AA,
        )
        panels.append(panel)

    # Assemble the 2×2 grid.
    top    = np.hstack([panels[0], panels[1]])
    bottom = np.hstack([panels[2], panels[3]])
    grid   = np.vstack([top, bottom])
    return grid


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        SUMMARY TABLE                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def print_summary(records: Dict[str, Dict]) -> None:
    """
    Print a formatted console table.
    records = { model_name: { "times": [ms, ...], "hits": [bool, ...] } }
    """
    header = f"{'Model':<20} {'Avg ms':>8} {'Avg FPS':>9} {'Success':>9} {'Hit/Total':>11}"
    sep    = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print("  FACE DETECTION BENCHMARK RESULTS")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)

    for name in MODEL_NAMES:
        data   = records[name]
        times  = data["times"]
        hits   = data["hits"]

        if not times:
            print(f"{name:<20} {'N/A':>8} {'N/A':>9} {'N/A':>9} {'0/0':>11}")
            continue

        avg_ms   = sum(times) / len(times)
        avg_fps  = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        n_hits   = sum(hits)
        n_total  = len(hits)
        pct      = (n_hits / n_total * 100) if n_total > 0 else 0.0

        print(
            f"{name:<20} {avg_ms:>7.1f}ms {avg_fps:>8.1f} "
            f"{pct:>8.1f}% {n_hits:>4}/{n_total:<4}"
        )

    print(sep + "\n")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                      PERFORMANCE GRAPH                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def save_performance_graph(records: Dict[str, Dict], output_dir: str) -> None:
    """
    Generate a dual Y-axis chart:
        • Primary axis   – Average FPS per model (bar chart)
        • Secondary axis  – Detection Success Rate % (line chart with markers)

    The figure is saved as a timestamped PNG inside benchmark_graphs/
    (created alongside this script).  plt.show() is NOT called so the
    script stays fully automated.
    """
    # ── 1. Extract metrics from the live records dict ─────────────────
    names       = []
    avg_fps_lst = []
    success_pct = []

    for name in MODEL_NAMES:
        data  = records[name]
        times = data["times"]
        hits  = data["hits"]

        # Skip models that failed to load (no data collected).
        if len(times) == 0:
            continue

        avg_ms  = sum(times) / len(times)
        avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        n_hits  = sum(hits)
        n_total = len(hits)
        pct     = (n_hits / n_total * 100) if n_total > 0 else 0.0

        names.append(name)
        avg_fps_lst.append(round(avg_fps, 1))
        success_pct.append(round(pct, 1))

    if not names:
        print("[Graph] No model data to plot.")
        return

    # ── 2. Create the figure ──────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar positions
    x = np.arange(len(names))
    bar_width = 0.5

    # ── Primary Y-axis: Average FPS (bars) ────────────────────────────
    bar_colors = ["#3498db", "#f1c40f", "#2ecc71", "#9b59b6"][:len(names)]
    bars = ax1.bar(x, avg_fps_lst, width=bar_width, color=bar_colors,
                   edgecolor="white", linewidth=0.8, label="Avg FPS", zorder=2)

    ax1.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Average FPS", fontsize=12, fontweight="bold", color="#2c3e50")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#2c3e50")
    ax1.set_ylim(0, max(avg_fps_lst) * 1.35)   # headroom for labels

    # Text labels on bars
    for bar, val in zip(bars, avg_fps_lst):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val}", ha="center", va="bottom", fontsize=10, fontweight="bold",
            color="#2c3e50",
        )

    # ── Secondary Y-axis: Success Rate % (line) ──────────────────────
    ax2 = ax1.twinx()
    line_color = "#e74c3c"
    ax2.plot(
        x, success_pct, color=line_color, marker="o", markersize=9,
        linewidth=2.5, label="Success Rate %", zorder=3,
    )
    ax2.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold",
                   color=line_color)
    ax2.tick_params(axis="y", labelcolor=line_color)
    ax2.set_ylim(0, 110)   # 0-100% with some padding

    # Text labels on line points
    for xi, val in zip(x, success_pct):
        ax2.annotate(
            f"{val}%", (xi, val), textcoords="offset points",
            xytext=(0, 12), ha="center", fontsize=10, fontweight="bold",
            color=line_color,
        )

    # ── Title, legend, grid ───────────────────────────────────────────
    ax1.set_title(
        "Face Detection Benchmark: Speed vs. Accuracy",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # Combine legends from both axes into one box.
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2,
              loc="upper right", fontsize=10, framealpha=0.9)

    fig.tight_layout()

    # ── 3. Save with timestamp ────────────────────────────────────────
    graphs_dir = os.path.join(SCRIPT_DIR, "benchmark_graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_graph_{ts}.png"
    filepath = os.path.join(graphs_dir, filename)

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Graph] Performance chart saved → {filepath}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                             MAIN                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    # ── CLI args ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Face Detection Model Benchmark")
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT_DIR,
        help="Folder of test images (default: captured_targets/)",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_DIR,
        help="Folder to save 2×2 grids (default: benchmark_results/)",
    )
    args = parser.parse_args()

    input_dir  = args.input
    output_dir = args.output

    # ── Collect images ────────────────────────────────────────────────
    exts   = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(input_dir, ext)))
    images.sort()

    if not images:
        print(f"[Benchmark] No images found in: {input_dir}")
        print("  Run main_watchdog.py first to populate captured_targets/.")
        sys.exit(1)

    print(f"[Benchmark] Found {len(images)} test image(s) in: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # ── Initialise models ─────────────────────────────────────────────

    # 1. Haar Cascade
    haar_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if haar_cascade.empty():
        print("[Benchmark] WARNING: Haar cascade XML failed to load.")
    else:
        print("[Benchmark] ✓ Haar Cascade loaded.")

    # 2. YuNet
    yunet_detector = None
    if os.path.isfile(YUNET_MODEL_PATH):
        yunet_detector = cv2.FaceDetectorYN.create(
            model=YUNET_MODEL_PATH,
            config="",
            input_size=(320, 320),        # Will be resized per-image
            score_threshold=0.7,
            nms_threshold=0.3,
            top_k=10,
        )
        print("[Benchmark] ✓ YuNet loaded.")
    else:
        print(
            f"[Benchmark] WARNING: YuNet ONNX not found at:\n"
            f"  {YUNET_MODEL_PATH}\n"
            f"  Download from:\n"
            f"  https://github.com/opencv/opencv_zoo/blob/main/models/"
            f"face_detection_yunet/face_detection_yunet_2023mar.onnx"
        )

    # 3. MTCNN
    mtcnn_detector = None
    try:
        from mtcnn import MTCNN
        mtcnn_detector = MTCNN()
        print("[Benchmark] ✓ MTCNN loaded.")
    except ImportError:
        print("[Benchmark] WARNING: mtcnn not installed.  pip install mtcnn")

    # 4. MediaPipe
    mp_face = None
    try:
        import mediapipe as mp
        mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=0,           # 0 = short-range (< 2 m), good for crops
            min_detection_confidence=0.5,
        )
        print("[Benchmark] ✓ MediaPipe Face Detection loaded.")
    except ImportError:
        print("[Benchmark] WARNING: mediapipe not installed.  pip install mediapipe")

    # ── Metrics accumulators ──────────────────────────────────────────
    records: Dict[str, Dict] = {
        name: {"times": [], "hits": []} for name in MODEL_NAMES
    }

    # ── Process every image ───────────────────────────────────────────
    print(f"\n{'─' * 60}")
    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [{idx+1}] Skipping unreadable file: {img_path}")
            continue

        basename = os.path.basename(img_path)
        print(f"  [{idx+1}/{len(images)}] {basename}  ({img.shape[1]}×{img.shape[0]})")

        all_faces: List[List] = []
        times_ms:  List[float] = []

        # ── Run each model ────────────────────────────────────────────

        # 1. Haar
        faces_haar, t_haar = detect_haar(img, haar_cascade)
        all_faces.append(faces_haar)
        times_ms.append(t_haar)
        records["Haar Cascade"]["times"].append(t_haar)
        records["Haar Cascade"]["hits"].append(len(faces_haar) > 0)

        # 2. YuNet
        faces_yunet, t_yunet = detect_yunet(img, yunet_detector)
        all_faces.append(faces_yunet)
        times_ms.append(t_yunet)
        records["YuNet (DNN)"]["times"].append(t_yunet)
        records["YuNet (DNN)"]["hits"].append(len(faces_yunet) > 0)

        # 3. MTCNN
        if mtcnn_detector is not None:
            faces_mtcnn, t_mtcnn = detect_mtcnn(img, mtcnn_detector)
        else:
            faces_mtcnn, t_mtcnn = [], 0.0
        all_faces.append(faces_mtcnn)
        times_ms.append(t_mtcnn)
        records["MTCNN"]["times"].append(t_mtcnn)
        records["MTCNN"]["hits"].append(len(faces_mtcnn) > 0)

        # 4. MediaPipe
        if mp_face is not None:
            faces_mp, t_mp = detect_mediapipe(img, mp_face)
        else:
            faces_mp, t_mp = [], 0.0
        all_faces.append(faces_mp)
        times_ms.append(t_mp)
        records["MediaPipe"]["times"].append(t_mp)
        records["MediaPipe"]["hits"].append(len(faces_mp) > 0)

        # ── Build & save 2×2 grid ────────────────────────────────────
        grid = build_grid(img, all_faces, MODEL_NAMES, MODEL_COLORS, times_ms)
        out_name = f"grid_{os.path.splitext(basename)[0]}.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, grid)

    print(f"{'─' * 60}")
    print(f"\n[Benchmark] 2×2 grids saved to: {output_dir}")

    # ── Clean up MediaPipe ────────────────────────────────────────────
    if mp_face is not None:
        mp_face.close()

    # ── Summary table ─────────────────────────────────────────────────
    print_summary(records)

    # ── Performance graph ─────────────────────────────────────────────
    save_performance_graph(records, output_dir)


if __name__ == "__main__":
    main()
