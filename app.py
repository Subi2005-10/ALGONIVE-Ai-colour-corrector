"""
AI-Powered Image Colorizer
Streamlit web application for colorizing black & white images
using OpenCV's pre-trained deep learning model.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os
import time
from pathlib import Path
# Auto-download models if missing
import urllib.request
from pathlib import Path

def download_models():
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)
    
    files = {
        "colorization_deploy_v2.prototxt": "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt",
        "pts_in_hull.npy": "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy",
    }
    
    for filename, url in files.items():
        dest = MODEL_DIR / filename
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)

download_models()

from utils.colorizer import ImageColorizer
from utils.image_utils import (
    load_image,
    pil_to_cv2,
    cv2_to_pil,
    ensure_grayscale,
    resize_for_display,
    get_image_stats,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Image Colorizer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        color: #155724;
    }
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 1rem;
        color: #0c5460;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-size: 1rem;
        font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.9;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">🎨 AI Image Colorizer</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Transform black & white photos into vibrant color images using deep learning</p>',
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Model")
    model_choice = st.selectbox(
        "Colorization model",
        ["Zhang et al. (Caffe)", "Eccv16 (Legacy)"],
        help="Zhang et al. produces more vivid colors; Eccv16 is faster but more conservative.",
    )

    st.subheader("Post-processing")
    saturation = st.slider("Saturation boost", 0.5, 2.0, 1.2, 0.1,
                           help="Increase color vividness after colorization.")
    sharpness = st.slider("Sharpness", 0.0, 2.0, 1.0, 0.1,
                          help="Sharpen the final output image.")

    st.subheader("Output")
    output_size = st.selectbox("Max output size", ["Original", "1024px", "800px", "512px"])
    output_format = st.selectbox("Download format", ["PNG", "JPEG", "WEBP"])

    st.divider()
    st.markdown("### 📖 About")
    st.markdown("""
This app uses a **Caffe deep learning model** trained on millions of images to predict
realistic colors for grayscale photos.

**Model**: [Zhang et al., ECCV 2016](https://richzhang.github.io/colorization/)

**Tech stack**: Python · OpenCV DNN · Streamlit · NumPy · Pillow
    """)

# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_colorizer():
    """Load and cache the colorization model."""
    colorizer = ImageColorizer()
    colorizer.load_model()
    return colorizer

# ── Main layout ───────────────────────────────────────────────────────────────
col_upload, col_spacer, col_result = st.columns([1, 0.05, 1])

with col_upload:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a black & white image",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        help="Upload a grayscale or black & white photograph.",
    )

    use_sample = st.checkbox("Use a sample image instead")
    sample_path = None
    if use_sample:
        sample_dir = Path("sample_images")
        samples = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
        if samples:
            selected = st.selectbox("Choose sample", [s.name for s in samples])
            sample_path = sample_dir / selected
        else:
            st.info("No sample images found. Add images to `sample_images/` folder.")

# ── Process when image available ──────────────────────────────────────────────
image_source = uploaded_file or sample_path

if image_source:
    # Load image
    if uploaded_file:
        pil_img = Image.open(uploaded_file).convert("RGB")
    else:
        pil_img = Image.open(sample_path).convert("RGB")

    # Show original in left column
    with col_upload:
        st.image(pil_img, caption="Original image", use_container_width=True)

        stats = get_image_stats(pil_img)
        c1, c2 = st.columns(2)
        c1.metric("Width", f"{stats['width']}px")
        c2.metric("Height", f"{stats['height']}px")
        c1.metric("Mode", stats["mode"])
        c2.metric("Size", stats["size_kb"])

        colorize_btn = st.button("🎨 Colorize Image", type="primary")

    # Colorize
    if colorize_btn:
        with col_result:
            st.subheader("🖼️ Colorized Output")
            with st.spinner("Loading model and colorizing…"):
                try:
                    colorizer = load_colorizer()

                    # Convert to grayscale if needed, then colorize
                    gray_img = ensure_grayscale(pil_img)
                    cv2_gray = pil_to_cv2(gray_img)

                    start = time.time()
                    colorized_cv2 = colorizer.colorize(
                        cv2_gray,
                        saturation=saturation,
                        sharpness=sharpness,
                    )
                    elapsed = time.time() - start

                    colorized_pil = cv2_to_pil(colorized_cv2)

                    # Resize if requested
                    max_px = {
                        "Original": None,
                        "1024px": 1024,
                        "800px": 800,
                        "512px": 512,
                    }[output_size]
                    if max_px:
                        colorized_pil = resize_for_display(colorized_pil, max_px)

                    st.image(colorized_pil, caption="Colorized result", use_container_width=True)

                    # Stats
                    st.markdown(
                        f'<div class="success-box">✅ Colorized in <strong>{elapsed:.2f}s</strong></div>',
                        unsafe_allow_html=True,
                    )

                    # Download button
                    fmt = output_format.lower()
                    mime = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}[fmt]
                    buf = io.BytesIO()
                    colorized_pil.save(buf, format=output_format)
                    buf.seek(0)

                    st.download_button(
                        label=f"⬇️ Download {output_format}",
                        data=buf,
                        file_name=f"colorized.{fmt}",
                        mime=mime,
                    )

                    # Side-by-side comparison
                    st.divider()
                    st.subheader("🔄 Before / After")
                    bc1, bc2 = st.columns(2)
                    bc1.image(pil_img, caption="Original", use_container_width=True)
                    bc2.image(colorized_pil, caption="Colorized", use_container_width=True)

                except FileNotFoundError as e:
                    st.error(f"Model files not found: {e}\n\nRun `python setup_models.py` to download them.")
                except Exception as e:
                    st.error(f"Colorization failed: {e}")
                    st.exception(e)

else:
    # Placeholder state
    with col_result:
        st.subheader("🖼️ Colorized Output")
        st.markdown(
            '<div class="info-box">👈 Upload an image and click <strong>Colorize</strong> to see the result here.</div>',
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#999; font-size:0.85rem;">
    Built with Streamlit · OpenCV DNN · Zhang et al. ECCV 2016 colorization model
</div>
""", unsafe_allow_html=True)
