#!/usr/bin/env python3
"""
cli.py — Command-line interface for batch image colorization.

Usage examples
--------------
Colorize a single image:
    python cli.py photo.jpg

Colorize with options:
    python cli.py photo.jpg --output colorized.png --saturation 1.5 --sharpness 1.2

Batch colorize a folder:
    python cli.py ./old_photos/ --output ./colorized/ --batch

Show help:
    python cli.py --help
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

from utils.colorizer import ImageColorizer
from utils.image_utils import calculate_colorfulness

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="🎨 AI Image Colorizer — Zhang et al. ECCV 2016",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Input image file or directory (with --batch)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file or directory (default: <input>_colorized.<ext>)")
    parser.add_argument("--saturation", type=float, default=1.2,
                        help="Saturation multiplier (default: 1.2)")
    parser.add_argument("--sharpness", type=float, default=1.0,
                        help="Sharpness amount (default: 1.0)")
    parser.add_argument("--batch", action="store_true",
                        help="Treat input as a directory and colorize all images inside")
    parser.add_argument("--stats", action="store_true",
                        help="Print colorfulness score for each image")
    return parser.parse_args()


def colorize_one(colorizer, src: Path, dst: Path, args):
    img = cv2.imread(str(src))
    if img is None:
        print(f"  ⚠️  Cannot read {src.name} — skipping")
        return

    t0 = time.time()
    result = colorizer.colorize(img, saturation=args.saturation, sharpness=args.sharpness)
    elapsed = time.time() - t0

    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), result)

    msg = f"  ✅  {src.name} → {dst.name}  ({elapsed:.2f}s)"
    if args.stats:
        score = calculate_colorfulness(result)
        msg += f"  colorfulness={score:.1f}"
    print(msg)


def main():
    args = parse_args()
    src = Path(args.input)

    if not src.exists():
        print(f"❌ Input not found: {src}")
        sys.exit(1)

    print("\n🎨 AI Image Colorizer")
    print("Loading model…")
    colorizer = ImageColorizer()
    colorizer.load_model()
    print("Model ready.\n")

    if args.batch or src.is_dir():
        if not src.is_dir():
            print(f"❌ {src} is not a directory. Remove --batch or point to a folder.")
            sys.exit(1)
        images = [p for p in src.iterdir() if p.suffix.lower() in SUPPORTED]
        if not images:
            print(f"No supported images found in {src}")
            sys.exit(0)

        out_dir = Path(args.output) if args.output else src.parent / (src.name + "_colorized")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Batch colorizing {len(images)} image(s) → {out_dir}\n")

        for img_path in sorted(images):
            dst = out_dir / img_path.name
            colorize_one(colorizer, img_path, dst, args)
    else:
        # Single file
        if args.output:
            dst = Path(args.output)
        else:
            dst = src.parent / (src.stem + "_colorized" + src.suffix)
        print(f"Colorizing {src.name}…\n")
        colorize_one(colorizer, src, dst, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
