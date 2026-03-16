import os
import sys
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"

FILES = {
    "colorization_deploy_v2.prototxt": (
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/"
        "colorization/models/colorization_deploy_v2.prototxt"
    ),
    "colorization_release_v2.caffemodel": (
        "https://huggingface.co/datasets/howchihlee/colorization/"
        "resolve/main/colorization_release_v2.caffemodel"
    ),
    
    "pts_in_hull.npy": (
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/"
        "colorization/resources/pts_in_hull.npy"
    ),
}


def download(url: str, dest: Path) -> None:
    print(f"  Downloading {dest.name} …", end=" ", flush=True)

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"\r  [{bar}] {pct:5.1f}%  {dest.name}", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print(f"\r  ✅  {dest.name} saved ({dest.stat().st_size / 1024:.1f} KB)")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n🎨 AI Image Colorizer — Model Setup")
    print("=" * 45)

    all_present = True
    for filename, url in FILES.items():
        dest = MODEL_DIR / filename
        if dest.exists():
            print(f"  ✅  {filename} already present ({dest.stat().st_size / 1024:.1f} KB)")
        else:
            all_present = False
            try:
                download(url, dest)
            except Exception as e:
                print(f"\n  ❌  Failed to download {filename}: {e}")
                print(f"     URL: {url}")
                sys.exit(1)

    print("\n" + "=" * 45)
    if all_present:
        print("✅  All model files are already present.")
    else:
        print("✅  Setup complete. Run the app with:")
        print("    streamlit run app.py\n")


if __name__ == "__main__":
    main()