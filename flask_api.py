"""
flask_api.py — REST API backend for the colorizer.

Endpoints
---------
POST /api/colorize
    Body: multipart/form-data with field `image`
    Optional query params: saturation, sharpness, format (png|jpeg|webp)
    Returns: colorized image bytes

GET /api/health
    Returns: {"status": "ok", "model_loaded": true}

Run:
    python flask_api.py
"""

import io
import time
import traceback

from flask import Flask, request, jsonify, send_file
from PIL import Image
import numpy as np
import cv2

from utils.colorizer import ImageColorizer
from utils.image_utils import pil_to_cv2, cv2_to_pil, ensure_grayscale

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload limit

# Load model once at startup
colorizer = ImageColorizer()
_model_loaded = False


def get_colorizer():
    global _model_loaded
    if not _model_loaded:
        colorizer.load_model()
        _model_loaded = True
    return colorizer


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": _model_loaded})


@app.route("/api/colorize", methods=["POST"])
def colorize():
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    saturation = float(request.args.get("saturation", 1.2))
    sharpness  = float(request.args.get("sharpness",  1.0))
    fmt        = request.args.get("format", "png").lower()

    mime_map = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}
    if fmt not in mime_map:
        return jsonify({"error": f"Unsupported format: {fmt}"}), 400

    try:
        pil_img   = Image.open(file.stream).convert("RGB")
        gray_pil  = ensure_grayscale(pil_img)
        cv2_img   = pil_to_cv2(gray_pil)

        t0       = time.time()
        result   = get_colorizer().colorize(cv2_img, saturation=saturation, sharpness=sharpness)
        elapsed  = time.time() - t0

        result_pil = cv2_to_pil(result)
        buf = io.BytesIO()
        result_pil.save(buf, format=fmt.upper() if fmt != "jpeg" else "JPEG")
        buf.seek(0)

        response = send_file(buf, mimetype=mime_map[fmt])
        response.headers["X-Colorize-Time"] = f"{elapsed:.3f}s"
        return response

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🎨 AI Image Colorizer — Flask API")
    print("Pre-loading model…")
    try:
        get_colorizer()
        print("Model loaded. Starting server on http://localhost:5000\n")
    except FileNotFoundError:
        print("⚠️  Model files missing. Run: python setup_models.py")
        print("Starting server anyway (colorize endpoint will return 503).\n")

    app.run(debug=False, host="0.0.0.0", port=5000)
