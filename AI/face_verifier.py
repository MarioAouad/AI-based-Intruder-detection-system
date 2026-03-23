"""
face_verifier.py  –  YuNet-Based Face Verification Module
==========================================================

Follows the same class pattern as spatial_math.py:
  - Loads a model at __init__ time (YuNet ONNX via OpenCV DNN).
  - Exposes a single primary method: verify(crop_img) → bool.
  - Gracefully degrades if the .onnx file is missing.

The YuNet detector is a lightweight CNN trained for face detection.
It is used here as a binary gate: "Does this cropped torso image
contain a recognisable face?"

Download the ONNX weights from:
  https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

Place the file in the same directory as this script.
"""

import os
from typing import Optional

import cv2
import numpy as np

# Path to YuNet ONNX model – expected alongside this file.
_SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
_YUNET_ONNX_PATH = os.path.join(_SCRIPT_DIR, "face_detection_yunet_2023mar.onnx")


class FaceVerifier:
    """
    Lightweight face presence check using OpenCV's YuNet detector.

    Usage
    -----
    verifier = FaceVerifier()
    has_face = verifier.verify(cropped_bgr_image)

    Parameters
    ----------
    model_path : str
        Path to the YuNet ONNX weight file.
    score_threshold : float
        Minimum confidence for a detection to count as a face.
    """

    def __init__(
        self,
        model_path: str      = _YUNET_ONNX_PATH,
        score_threshold: float = 0.7,
    ):
        self.detector: Optional[cv2.FaceDetectorYN] = None

        if not os.path.isfile(model_path):
            print(
                f"[FaceVerifier] ERROR: YuNet ONNX file not found at:\n"
                f"  {model_path}\n"
                f"  Download from:\n"
                f"  https://github.com/opencv/opencv_zoo/blob/main/models/"
                f"face_detection_yunet/face_detection_yunet_2023mar.onnx\n"
                f"  Face verification will be DISABLED (verify() returns False)."
            )
            return

        # Initialise the detector with a dummy input size.
        # setInputSize() is called per-crop inside verify() to match the
        # actual image dimensions (YuNet requires this).
        self.detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),      # Placeholder; overridden per-call
            score_threshold=score_threshold,
            nms_threshold=0.3,
            top_k=5,
        )
        print(f"[FaceVerifier] YuNet loaded (score_threshold={score_threshold}).")

    # ------------------------------------------------------------------
    # Primary method
    # ------------------------------------------------------------------

    def verify(self, crop_img: np.ndarray) -> bool:
        """
        Check whether a face is present in the given BGR image crop.

        Parameters
        ----------
        crop_img : np.ndarray
            BGR image (typically a bounding-box crop from main_watchdog).

        Returns
        -------
        bool
            True if at least one face is detected above the score threshold,
            False otherwise (including when the model failed to load).
        """
        if self.detector is None:
            return False  # Model not loaded — graceful degradation

        if crop_img is None or crop_img.size == 0:
            return False

        h, w = crop_img.shape[:2]

        # YuNet requires setInputSize() to match the input dimensions
        # before every detect() call; mismatched sizes cause silent failures.
        self.detector.setInputSize((w, h))

        _, detections = self.detector.detect(crop_img)

        # detections is None when no faces are found.
        return detections is not None and len(detections) > 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        """Return True if the YuNet model was successfully initialised."""
        return self.detector is not None
