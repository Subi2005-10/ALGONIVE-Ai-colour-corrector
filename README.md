<<<<<<< HEAD
# 🎨 AI Image Colorizer

Automatically colorize black & white photographs using a **pre-trained deep learning model** (Zhang et al., ECCV 2016). Built with Python, OpenCV DNN, and Streamlit.

---

## ✨ Features

| Feature | Details |
|---|---|
| Deep learning colorization | Zhang et al. ECCV 2016 Caffe model |
| Web UI | Streamlit with drag-and-drop upload |
| REST API | Flask endpoint for programmatic use |
| CLI | Batch-process entire folders |
| Post-processing | Saturation boost + unsharp sharpening |
| Output formats | PNG · JPEG · WebP |
| Download | One-click colorized image download |

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/your-repo/ai-image-colorizer.git
cd ai-image-colorizer
pip install -r requirements.txt
```

### 2. Download the pre-trained model (~125 MB)

```bash
python setup_models.py
```

### 3. Launch the Streamlit web app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📁 Project Structure

```
ai-image-colorizer/
│
├── app.py                  # Streamlit web application (main entry point)
├── flask_api.py            # REST API backend
├── cli.py                  # Command-line interface
├── setup_models.py         # Model download script
├── requirements.txt        # Python dependencies
│
├── utils/
│   ├── __init__.py
│   ├── colorizer.py        # Core ImageColorizer class
│   └── image_utils.py      # Format conversion, stats, preprocessing
│
├── models/                 # Downloaded model files (created by setup_models.py)
│   ├── colorization_deploy_v2.prototxt
│   ├── colorization_release_v2.caffemodel
│   └── pts_in_hull.npy
│
├── sample_images/          # Place sample B&W images here
├── uploads/                # Temporary upload storage
└── outputs/                # Colorized image outputs
```

---

## 🖥️ Usage

### Web UI (Streamlit)

```bash
streamlit run app.py
```

1. Upload a grayscale / black & white image (JPG, PNG, BMP, TIFF, WebP)
2. Adjust saturation and sharpness in the sidebar
3. Click **Colorize Image**
4. Download the colorized result

### Command-Line Interface

```bash
# Colorize a single image
python cli.py old_photo.jpg

# Colorize with options
python cli.py old_photo.jpg --output result.png --saturation 1.5 --sharpness 1.2

# Batch colorize a folder
python cli.py ./old_photos/ --batch --output ./colorized/

# Show colorfulness metric
python cli.py old_photo.jpg --stats
```

### REST API (Flask)

```bash
# Start the API server
python flask_api.py

# Colorize via curl
curl -X POST http://localhost:5000/api/colorize \
     -F "image=@photo.jpg" \
     --output colorized.png

# With options
curl -X POST "http://localhost:5000/api/colorize?saturation=1.5&format=jpeg" \
     -F "image=@photo.jpg" \
     --output colorized.jpg

# Health check
curl http://localhost:5000/api/health
```

### Python API

```python
from utils.colorizer import ImageColorizer
import cv2

colorizer = ImageColorizer()
colorizer.load_model()

# Colorize a single image
img = cv2.imread("old_photo.jpg")
colorized = colorizer.colorize(img, saturation=1.3, sharpness=1.0)
cv2.imwrite("colorized.jpg", colorized)

# Or use the convenience method
colorizer.colorize_file("old_photo.jpg", "colorized.jpg", saturation=1.3)
```

---

## 🧠 How It Works

```
Input (B&W image)
        │
        ▼
 Convert to LAB color space
        │
        ▼
 Extract L channel (lightness only)
        │
        ▼
 Resize to 224×224 and feed to CNN
        │
        ▼
 Network predicts ab channels (313 color bins)
        │
        ▼
 Upsample ab back to original size
        │
        ▼
 Merge original L + predicted ab
        │
        ▼
 Convert LAB → RGB → Output
```

The model uses a **classification approach** rather than regression — it predicts probability distributions over 313 quantized color bins, avoiding the "muddy grey" problem common with naive regression models.

**Reference**: Zhang, R., Isola, P., & Efros, A. A. (2016). *Colorful Image Colorization*. ECCV 2016.
[Paper](https://arxiv.org/abs/1603.08511) | [Project page](https://richzhang.github.io/colorization/)

---

## ⚙️ Configuration

| Setting | Default | Description |
|---|---|---|
| `saturation` | 1.2 | Color saturation multiplier (1.0 = neutral) |
| `sharpness` | 1.0 | Unsharp-mask sharpening amount |
| `output_format` | PNG | Output image format |
| `max_output_size` | Original | Resize longest side to N pixels |

---

## 🛠️ Requirements

- Python 3.9+
- opencv-python ≥ 4.8
- numpy ≥ 1.24
- Pillow ≥ 10.0
- streamlit ≥ 1.35 (web UI)
- flask ≥ 3.0 (REST API)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

Model weights are from the [richzhang/colorization](https://github.com/richzhang/colorization) repository
and are subject to their respective license.
=======
# ai-colour-corrector
AI Color Corrector is a Python-based application that uses OpenCV and deep learning to automatically restore and enhance colors in grayscale or faded images. The model predicts realistic color channels to generate natural-looking results, making it useful for restoring old photos and experimenting with AI-based image processing.
>>>>>>> 699bfe4d15ed899fa8f2e625f28a567aff74e571
