"""
Image utility helpers — format conversion, stats, preprocessing.
"""

from __future__ import annotations

import io
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance


# ── Format conversions ────────────────────────────────────────────────────────

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL Image (RGB) to an OpenCV ndarray (BGR)."""
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR ndarray to a PIL Image (RGB)."""
    rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_image(source) -> Image.Image:
    """
    Load an image from a file path, bytes, or file-like object.

    Returns a PIL Image in RGB mode.
    """
    if isinstance(source, (str,)):
        return Image.open(source).convert("RGB")
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source)).convert("RGB")
    # file-like (Streamlit UploadedFile, BytesIO, etc.)
    return Image.open(source).convert("RGB")


# ── Preprocessing ─────────────────────────────────────────────────────────────

def ensure_grayscale(pil_img: Image.Image) -> Image.Image:
    """
    Return a grayscale version of the image converted back to RGB.

    The network expects an image whose color information has been stripped —
    we convert to L (luminance) then back to RGB so OpenCV can read it.
    """
    return pil_img.convert("L").convert("RGB")


def resize_for_display(pil_img: Image.Image, max_px: int) -> Image.Image:
    """Resize image so its longest side is ≤ max_px (maintains aspect ratio)."""
    w, h = pil_img.size
    longest = max(w, h)
    if longest <= max_px:
        return pil_img
    scale = max_px / longest
    new_w, new_h = int(w * scale), int(h * scale)
    return pil_img.resize((new_w, new_h), Image.LANCZOS)


# ── Image stats ───────────────────────────────────────────────────────────────

def get_image_stats(pil_img: Image.Image) -> dict:
    """Return a dict of display-friendly image metadata."""
    w, h = pil_img.size
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    size_kb = len(buf.getvalue()) / 1024

    return {
        "width": w,
        "height": h,
        "mode": pil_img.mode,
        "size_kb": f"{size_kb:.1f} KB",
        "megapixels": f"{(w * h) / 1_000_000:.2f} MP",
    }


# ── Histogram comparison ──────────────────────────────────────────────────────

def color_histogram(cv2_img: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-channel histograms (B, G, R)."""
    hists = {}
    for i, ch in enumerate(["B", "G", "R"]):
        hists[ch] = cv2.calcHist([cv2_img], [i], None, [256], [0, 256]).flatten()
    return hists


# ── Quality helpers ───────────────────────────────────────────────────────────

def calculate_colorfulness(cv2_img: np.ndarray) -> float:
    """
    Hasler & Süsstrunk (2003) colorfulness metric.
    Higher → more colorful.
    """
    b, g, r = cv2.split(cv2_img.astype(np.float32))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg, mean_rg = rg.std(), rg.mean()
    std_yb, mean_yb = yb.std(), yb.mean()
    std_root  = np.sqrt(std_rg**2 + std_yb**2)
    mean_root = np.sqrt(mean_rg**2 + mean_yb**2)
    return std_root + 0.3 * mean_root
