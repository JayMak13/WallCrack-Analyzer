"""Physical image calibration using a printed ArUco marker.

An uncalibrated photo has no reliable pixels-to-millimetres conversion.  This
module detects a square marker of known printed side length in the same plane as
the crack and returns a scale for that specific photo.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = hasattr(cv2, "aruco")
except ImportError:
    OPENCV_AVAILABLE = False


MARKER_ID = 0
MARKER_DICT = "DICT_4X4_50"


@dataclass(frozen=True)
class Calibration:
    mm_per_pixel: Optional[float]
    marker_side_pixels: Optional[float]
    message: str

    @property
    def available(self) -> bool:
        return self.mm_per_pixel is not None


def detect_scale(image_path: str | Path, marker_side_mm: float) -> Calibration:
    """Detect the marker and calculate scale; never infer a scale without it."""
    if not OPENCV_AVAILABLE:
        return Calibration(None, None, "Calibration is unavailable because OpenCV is not installed.")
    if marker_side_mm <= 0:
        return Calibration(None, None, "Marker side length must be greater than zero.")
    image = cv2.imread(str(image_path))
    if image is None:
        return Calibration(None, None, "The uploaded image could not be read for calibration.")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, MARKER_DICT))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return Calibration(None, None, "No calibration marker found. Add the printed marker beside the crack to measure it.")
    for detected_corners, detected_id in zip(corners, ids.flatten()):
        if int(detected_id) == MARKER_ID:
            points = detected_corners.reshape(4, 2).astype(float)
            sides = [np.linalg.norm(points[(index + 1) % 4] - points[index]) for index in range(4)]
            pixels = float(np.mean(sides))
            if pixels < 20:
                return Calibration(None, None, "Marker is too small in the image; move closer and keep it in focus.")
            return Calibration(marker_side_mm / pixels, pixels, "Calibration marker detected.")
    return Calibration(None, None, "The expected calibration marker was not found.")


def create_marker_png(size_pixels: int = 800) -> Optional[bytes]:
    """Create a printable marker. Print it at the side length entered in the app."""
    if not OPENCV_AVAILABLE:
        return None
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, MARKER_DICT))
    marker = cv2.aruco.generateImageMarker(dictionary, MARKER_ID, size_pixels)
    success, encoded = cv2.imencode(".png", marker)
    return encoded.tobytes() if success else None
