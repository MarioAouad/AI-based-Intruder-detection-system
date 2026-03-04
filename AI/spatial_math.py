"""
spatial_math.py  –  Triangle Similarity Distance Engine
========================================================

Triangle Similarity (also called "similar triangles" or "focal-length method")
is a classic monocular depth estimation technique.

The core insight
----------------
When a camera photographs an object of known real-world width W at a distance D,
it appears to be P pixels wide in the image.  The relationship between these three
quantities and the camera's focal length F is:

        F = (P * D) / W          ← calibration step (D is fixed at 1 m)

Once F is known, the formula can be rearranged to compute D from any new P:

        D = (W * F) / P          ← live distance estimation

In this system:
    W  = REAL_TORSO_WIDTH_CM (in cm)
    D  = 100 cm  (1 metre, the calibration distance)
    P  = CALIBRATED_PIXEL_WIDTH_1M  (measured once by the user)
    F  = derived focal length  (stored in self.focal_length)
"""

from config import REAL_TORSO_WIDTH_CM, CALIBRATED_PIXEL_WIDTH_1M


class DistanceEstimator:
    """
    Computes real-world distance from the camera to a detected person using
    the Triangle Similarity method.

    Parameters
    ----------
    real_width_cm : float
        Known real-world width of the reference object (human torso/shoulders).
    pixel_width_at_1m : float or int
        How many pixels wide that same torso appears when the person stands
        exactly 1 metre from the camera.  Must be > 0.

    Raises
    ------
    RuntimeError
        If pixel_width_at_1m is 0 (calibration has not been performed yet).
    """

    def __init__(
        self,
        real_width_cm: float    = REAL_TORSO_WIDTH_CM,
        pixel_width_at_1m: float = CALIBRATED_PIXEL_WIDTH_1M,
    ):
        self.real_width_cm = real_width_cm

        if pixel_width_at_1m <= 0:
            # Focal length cannot be calculated without a valid calibration value.
            # We store 0 and let calculate_distance() return None gracefully.
            self.focal_length = 0.0
            print(
                "[DistanceEstimator] WARNING: CALIBRATED_PIXEL_WIDTH_1M = 0.\n"
                "  Distance estimation is DISABLED until you:\n"
                "    1. Stand exactly 1 m from your camera.\n"
                "    2. Note the pixel width of your shoulders in the bounding box.\n"
                "    3. Set CALIBRATED_PIXEL_WIDTH_1M in config.py and restart."
            )
        else:
            # Triangle Similarity: F = (P × D) / W
            # D is expressed in cm (100 cm = 1 m) so units are consistent.
            self.focal_length = (pixel_width_at_1m * 100.0) / real_width_cm
            print(
                f"[DistanceEstimator] Focal length computed: "
                f"{self.focal_length:.2f} px  "
                f"(ref: {pixel_width_at_1m} px @ 1 m, torso={real_width_cm} cm)"
            )

    def calculate_distance(self, pixel_width: float) -> float | None:
        """
        Estimate the person's distance from the camera in metres.

        Implements:  D = (W × F) / (P × 100)
        The division by 100 converts the result from centimetres to metres.

        Parameters
        ----------
        pixel_width : float
            Width (in pixels) of the detected person's bounding box.

        Returns
        -------
        float
            Estimated distance in metres, or None if calibration is missing
            or pixel_width is zero (prevents ZeroDivisionError).
        """
        if self.focal_length <= 0:
            return None  # Calibration not done yet

        if pixel_width <= 0:
            return None  # Guard: degenerate / collapsed bounding box

        # Triangle Similarity rearranged to solve for distance.
        distance_cm = (self.real_width_cm * self.focal_length) / pixel_width
        distance_m  = distance_cm / 100.0
        return round(distance_m, 2)

    def is_calibrated(self) -> bool:
        """Return True if a valid focal length has been calculated."""
        return self.focal_length > 0
