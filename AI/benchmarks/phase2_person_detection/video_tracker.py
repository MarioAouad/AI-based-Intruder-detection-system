"""
video_tracker.py
----------------
Core inference engine for person detection and tracking.

Responsibilities
----------------
  - Load an Ultralytics-compatible model (YOLO, RT-DETR) onto the GPU.
  - Run model.track() with ByteTrack (or BoTSORT) for persistent person IDs.
  - Draw annotated bounding boxes with sticky tracking IDs on each frame.
  - Overlay real-time FPS and inference latency (ms) in the top-left corner.
  - Support non-Ultralytics architectures (RF-DETR, RTMDet) via a plugin hook.

Design
------
The class follows a simple "process one frame at a time" API so that the
orchestrator can call it inside a standard frame loop and feed per-frame data
to BenchmarkLogger without tight coupling.
"""

import time
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch

from config import (
    DEVICE,
    HALF_PRECISION,
    TARGET_CLASS,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    TRACKER_CONFIG,
)


# ---------------------------------------------------------------------------
# Colour palette – one colour per tracker ID (mod 20 so it wraps for crowds)
# ---------------------------------------------------------------------------
_PALETTE = [
    (255,  56,  56), (255, 157,  51), (255, 178, 102), (230, 230,   0),
    (253,  85, 255), ( 99, 210, 240), ( 55, 234, 255), (  0, 154, 255),
    (  0, 255, 255), (  0, 255, 145), (  0, 255,  21), (100, 115, 255),
    (220, 130, 255), (  0, 255, 112), (130,   0, 255), (255,   0, 170),
    (255,  50,  50), (128, 255,   0), (255, 128,   0), (  0, 128, 255),
]


def _colour_for_id(track_id: int) -> Tuple[int, int, int]:
    """Return a deterministic BGR colour for a given track ID."""
    return _PALETTE[int(track_id) % len(_PALETTE)]


class VideoTracker:
    """
    Wraps an Ultralytics model and exposes a clean per-frame tracking interface.

    Parameters
    ----------
    model_cfg : dict
        Entry from config.MODELS, e.g.
        {"name": "YOLOv8-Nano", "weights": "yolov8n.pt", "type": "yolo"}
    """

    def __init__(self, model_cfg: dict):
        self.model_name  = model_cfg["name"]
        self.weights     = model_cfg["weights"]
        self.model_type  = model_cfg["type"]   # "yolo" | "rtdetr" | "rfdetr" | "rtmdet"

        # Timing state for FPS/latency calculation
        self._prev_tick: float = time.perf_counter()
        self._inf_latency_ms: float = 0.0
        self._fps: float = 0.0

        # The underlying model object (populated in _load_model)
        self.model = None

        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """
        Load the requested architecture onto the GPU.
        Ultralytics auto-downloads .pt files on first use if not cached.
        Non-Ultralytics backends are handled with dedicated helpers.
        """
        if self.model_type in ("yolo", "rtdetr"):
            self._load_ultralytics_model()
        elif self.model_type == "rfdetr":
            self._load_rfdetr_model()
        elif self.model_type == "rtmdet":
            self._load_rtmdet_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _load_ultralytics_model(self) -> None:
        """Load YOLO or RT-DETR model through the Ultralytics API."""
        from ultralytics import YOLO  # lazy import

        print(f"[VideoTracker] Loading '{self.model_name}' via Ultralytics …")
        self.model = YOLO(self.weights)  # downloads automatically if needed

        # Fuse conv+bn layers and switch to eval mode for maximum throughput.
        self.model.fuse()

        # Move to GPU; optionally use FP16 for speed (RTX 4060 handles it well).
        self.model.to(DEVICE)
        if HALF_PRECISION and DEVICE == "cuda":
            self.model.half()

        print(f"[VideoTracker] '{self.model_name}' ready on {DEVICE.upper()}.")

    def _load_rfdetr_model(self) -> None:
        """
        Load Roboflow DETR.
        Requires:  pip install rfdetr
        rfdetr does not expose a .track() method, so we run plain .predict()
        and handle identity assignment separately (IDs will all be 0,
        meaning the unique-ID metric measures detections, not tracks).
        """
        try:
            from rfdetr import RFDETRBase  # noqa: F401
            print("[VideoTracker] Loading RF-DETR (Roboflow DETR) …")
            self.model = RFDETRBase()
            self.model.model.to(DEVICE)
            print("[VideoTracker] RF-DETR loaded.")
        except ImportError:
            print("[VideoTracker] rfdetr not installed. Skipping RF-DETR. "
                  "Install with:  pip install rfdetr")
            self.model = None

    def _load_rtmdet_model(self) -> None:
        """
        Load RTMDet via MMDetection.
        Requires:  pip install mmdet mmcv-full openmim
                   mim install mmdet
        Uses the pretrained COCO checkpoint for 'rtmdet-tiny'.
        """
        try:
            from mmdet.apis import init_detector, inference_detector  # noqa: F401
            _config = "rtmdet_tiny_8xb32-300e_coco.py"
            _checkpoint = "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
            print("[VideoTracker] Loading RTMDet via MMDetection …")
            # Store the inference function as an attribute for later use.
            self._mmdet_infer = inference_detector
            self.model = init_detector(_config, _checkpoint, device=DEVICE)
            print("[VideoTracker] RTMDet loaded.")
        except ImportError:
            print("[VideoTracker] mmdet not installed. Skipping RTMDet. "
                  "Install with:  pip install mmdet")
            self.model = None

    # ------------------------------------------------------------------
    # Per-frame inference
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float, List[float], List[int]]:
        """
        Run detection + tracking on a single BGR frame.

        Returns
        -------
        annotated_frame : np.ndarray
            The input frame with bounding boxes, IDs, FPS and latency overlaid.
        fps : float
            Instantaneous frames per second for this call.
        confidences : list[float]
            Confidence scores of detected persons in this frame.
        track_ids : list[int]
            Tracker-assigned integer IDs for each detection.
        """
        if self.model is None:
            # Model failed to load (dependency missing); return blank frame.
            return frame, 0.0, [], []

        t_start = time.perf_counter()

        if self.model_type in ("yolo", "rtdetr"):
            annotated, confidences, track_ids = self._infer_ultralytics(frame)
        elif self.model_type == "rfdetr":
            annotated, confidences, track_ids = self._infer_rfdetr(frame)
        elif self.model_type == "rtmdet":
            annotated, confidences, track_ids = self._infer_rtmdet(frame)
        else:
            annotated, confidences, track_ids = frame.copy(), [], []

        # --- Timing ---------------------------------------------------------
        t_end = time.perf_counter()
        self._inf_latency_ms = (t_end - t_start) * 1_000        # ms
        elapsed_since_last   = t_end - self._prev_tick
        self._fps = 1.0 / elapsed_since_last if elapsed_since_last > 0 else 0.0
        self._prev_tick = t_end

        # Overlay HUD on the annotated frame.
        self._draw_hud(annotated)

        return annotated, self._fps, confidences, track_ids

    def _infer_ultralytics(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[float], List[int]]:
        """
        Run model.track() for Ultralytics models (YOLO / RT-DETR).

        persist=True is critical: it tells the tracker to keep state between
        consecutive frames, giving each person a stable ID across the video.
        """
        results = self.model.track(
            source=frame,
            persist=True,              # Keep tracker state across frames
            tracker=TRACKER_CONFIG,    # ByteTrack config bundled with Ultralytics
            classes=[TARGET_CLASS],    # Only detect persons (class 0)
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False,
            half=HALF_PRECISION and (DEVICE == "cuda"),
        )

        annotated   = frame.copy()
        confidences: List[float] = []
        track_ids:   List[int]   = []

        if results and results[0].boxes is not None:
            boxes  = results[0].boxes
            xyxys  = boxes.xyxy.cpu().numpy().astype(int)
            confs  = boxes.conf.cpu().numpy().tolist()
            ids    = boxes.id  # None if tracker hasn't assigned IDs yet

            for i, (x1, y1, x2, y2) in enumerate(xyxys):
                conf = confs[i]
                tid  = int(ids[i].item()) if (ids is not None and ids[i] is not None) else -1

                confidences.append(conf)
                if tid >= 0:
                    track_ids.append(tid)

                colour = _colour_for_id(tid) if tid >= 0 else (200, 200, 200)

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

                # Draw sticky label: "ID:N  conf%"
                label = f"ID:{tid}  {conf:.0%}" if tid >= 0 else f"? {conf:.0%}"
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )
                # Filled rectangle behind text for readability
                cv2.rectangle(
                    annotated,
                    (x1, y1 - th - baseline - 4),
                    (x1 + tw + 4, y1),
                    colour, -1
                )
                cv2.putText(
                    annotated, label,
                    (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA
                )

        return annotated, confidences, track_ids

    def _infer_rfdetr(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[float], List[int]]:
        """
        Inference for Roboflow DETR.
        rfdetr accepts a PIL Image; we convert, run, and parse outputs.
        """
        from PIL import Image  # PIL is a dep of rfdetr
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        detections = self.model.predict(pil_img)  # rfdetr Detections object
        annotated  = frame.copy()
        confidences: List[float] = []
        track_ids:   List[int]   = []

        # rfdetr uses supervision Detections; filter to person class.
        if detections is not None and len(detections) > 0:
            for i in range(len(detections.xyxy)):
                cls  = int(detections.class_id[i]) if detections.class_id is not None else -1
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                if cls != TARGET_CLASS or conf < CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                confidences.append(conf)
                # rfdetr has no built-in tracker; use detection index as proxy ID
                track_ids.append(i)
                colour = _colour_for_id(i)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                label = f"Person {conf:.0%}"
                cv2.putText(annotated, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

        return annotated, confidences, track_ids

    def _infer_rtmdet(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[float], List[int]]:
        """
        Inference for RTMDet via MMDetection.
        Returns boxes for COCO class 0 (person) above confidence threshold.
        """
        result = self._mmdet_infer(self.model, frame)
        annotated   = frame.copy()
        confidences: List[float] = []
        track_ids:   List[int]   = []

        # MMDetection returns a list of arrays, one per class.
        # Class 0 = person in COCO.
        person_boxes = result.pred_instances
        if person_boxes is not None:
            for i, (score, label) in enumerate(
                zip(person_boxes.scores.cpu(), person_boxes.labels.cpu())
            ):
                if int(label) != TARGET_CLASS:
                    continue
                conf = float(score)
                if conf < CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = person_boxes.bboxes[i].cpu().numpy().astype(int)
                confidences.append(conf)
                track_ids.append(i)
                colour = _colour_for_id(i)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                label_txt = f"Person {conf:.0%}"
                cv2.putText(annotated, label_txt, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

        return annotated, confidences, track_ids

    # ------------------------------------------------------------------
    # HUD overlay
    # ------------------------------------------------------------------

    def _draw_hud(self, frame: np.ndarray) -> None:
        """
        Overlay FPS and inference latency in the top-left corner of the frame.
        Background rectangle makes the text readable on any background.
        """
        lines = [
            f"Model : {self.model_name}",
            f"FPS   : {self._fps:.1f}",
            f"Latency: {self._inf_latency_ms:.1f} ms",
        ]
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness  = 1
        padding    = 6
        line_height = 20

        # Measure maximum text width for background box.
        max_w = max(cv2.getTextSize(l, font, font_scale, thickness)[0][0] for l in lines)
        box_h = len(lines) * line_height + padding * 2
        cv2.rectangle(frame, (0, 0), (max_w + padding * 2, box_h), (0, 0, 0), -1)

        for idx, line in enumerate(lines):
            y = padding + (idx + 1) * line_height - 4
            cv2.putText(frame, line, (padding, y),
                        font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Clean-up
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Delete the model and free GPU memory held by this tracker."""
        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[VideoTracker] '{self.model_name}' released from GPU.")
