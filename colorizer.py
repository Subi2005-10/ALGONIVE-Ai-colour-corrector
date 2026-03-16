"""
ImageColorizer — Core colorization engine
Uses OpenCV's DNN module with the Caffe model from Zhang et al. (ECCV 2016).

The model:
1. Receives a grayscale L-channel (from LAB color space)
2. Predicts ab-channels (color)
3. We merge L + ab → convert to RGB

Model files (download via setup_models.py):
  - models/colorization_deploy_v2.prototxt
  - models/colorization_release_v2.caffemodel
  - models/pts_in_hull.npy
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from pathlib import Path


MODEL_DIR = Path(__file__).parent.parent / "models"

PROTOTXT    = MODEL_DIR / "colorization_deploy_v2.prototxt"
CAFFEMODEL  = MODEL_DIR / "colorization_release_v2.caffemodel"
HULL_PTS    = MODEL_DIR / "pts_in_hull.npy"

# URLs for downloading model files
MODEL_URLS = {
    "prototxt": (
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/"
        "colorization/models/colorization_deploy_v2.prototxt"
    ),
    "caffemodel": (
        "http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/"
        "files/demo_v2/colorization_release_v2.caffemodel"
    ),
    "hull_pts": (
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/"
        "colorization/resources/pts_in_hull.npy"
    ),
}


class ImageColorizer:
    """
    Wraps the Zhang et al. ECCV 2016 colorization network.

    Usage
    -----
    >>> colorizer = ImageColorizer()
    >>> colorizer.load_model()
    >>> colorized = colorizer.colorize(gray_bgr_image)
    """

    def __init__(self):
        self.net = None
        self._model_loaded = False

    # ── Public API ─────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load the Caffe model into OpenCV DNN."""
        self._check_model_files()

        self.net = cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(CAFFEMODEL))

        # Load cluster centers and inject into the model layers
        pts = np.load(str(HULL_PTS))
        pts = pts.transpose().reshape(2, 313, 1, 1).astype(np.float32)

        class8  = self.net.getLayerId("class8_ab")
        conv8   = self.net.getLayerId("conv8_313_rh")

        self.net.getLayer(class8).blobs = [pts]
        self.net.getLayer(conv8).blobs  = [np.full([1, 313], 2.606, dtype=np.float32)]

        self._model_loaded = True

    def colorize(
        self,
        gray_bgr: np.ndarray,
        saturation: float = 1.2,
        sharpness: float = 1.0,
    ) -> np.ndarray:
        """
        Colorize a grayscale BGR image.

        Parameters
        ----------
        gray_bgr : np.ndarray
            Input image in BGR format (can be 1- or 3-channel).
        saturation : float
            Multiplier for color saturation (1.0 = no change).
        sharpness : float
            Unsharp-mask amount (1.0 = no sharpening).

        Returns
        -------
        np.ndarray
            Colorized image in BGR format, uint8.
        """
        if not self._model_loaded:
            self.load_model()

        # Ensure 3-channel BGR
        if gray_bgr.ndim == 2:
            gray_bgr = cv2.cvtColor(gray_bgr, cv2.COLOR_GRAY2BGR)
        elif gray_bgr.shape[2] == 4:
            gray_bgr = cv2.cvtColor(gray_bgr, cv2.COLOR_BGRA2BGR)

        # Convert to LAB and extract L channel
        img_lab = cv2.cvtColor(gray_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
        l_channel = img_lab[:, :, 0]

        # Resize L to 224×224 for the network
        l_resized = cv2.resize(l_channel, (224, 224))
        l_resized -= 50  # mean-center

        # Forward pass
        blob = cv2.dnn.blobFromImage(l_resized)
        self.net.setInput(blob)
        ab_pred = self.net.forward()[0, :, :, :].transpose(1, 2, 0)

        # Resize predicted ab back to original size
        h, w = gray_bgr.shape[:2]
        ab_resized = cv2.resize(ab_pred, (w, h))

        # Merge L (original) + predicted ab
        result_lab = np.concatenate(
            [l_channel[:, :, np.newaxis], ab_resized], axis=2
        )

        # Convert back to BGR uint8
        result_bgr = cv2.cvtColor(result_lab.astype(np.float32), cv2.COLOR_LAB2BGR)
        result_bgr = np.clip(result_bgr, 0, 1)
        result_bgr = (result_bgr * 255).astype(np.uint8)

        # Post-processing
        if saturation != 1.0:
            result_bgr = self._adjust_saturation(result_bgr, saturation)

        if sharpness != 1.0:
            result_bgr = self._adjust_sharpness(result_bgr, sharpness)

        return result_bgr

    # ── CLI convenience ────────────────────────────────────────────────────

    def colorize_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        **kwargs,
    ) -> None:
        """Colorize an image file and save to output_path."""
        img = cv2.imread(str(input_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {input_path}")
        result = self.colorize(img, **kwargs)
        cv2.imwrite(str(output_path), result)

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _check_model_files() -> None:
        missing = []
        for name, path in [("prototxt", PROTOTXT), ("caffemodel", CAFFEMODEL), ("hull_pts", HULL_PTS)]:
            if not path.exists():
                missing.append(str(path))
        if missing:
            raise FileNotFoundError(
                "Missing model files:\n  " + "\n  ".join(missing) +
                "\n\nRun: python setup_models.py"
            )

    @staticmethod
    def _adjust_saturation(img_bgr: np.ndarray, factor: float) -> np.ndarray:
        """Multiply the saturation channel in HSV space."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def _adjust_sharpness(img_bgr: np.ndarray, amount: float) -> np.ndarray:
        """Unsharp mask sharpening."""
        blurred = cv2.GaussianBlur(img_bgr, (0, 0), 3)
        return cv2.addWeighted(img_bgr, 1 + amount * 0.5, blurred, -amount * 0.5, 0)
