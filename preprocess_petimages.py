#!/usr/bin/env python3
"""Preprocess PetImages to 224x224 RGB for CNNs.

Usage examples:
  python scripts/preprocess_petimages.py \
    --input data/Dataset/PetImages \
    --output data/processed/PetImages_224 \
    --method resize_crop \
    --workers 8
"""
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
import concurrent.futures


def center_crop(img, size):
    w, h = img.size
    tw, th = size, size
    left = max(0, (w - tw) // 2)
    top = max(0, (h - th) // 2)
    return img.crop((left, top, left + tw, top + th))


def resize_shorter_side(img, target_short):
    w, h = img.size
    if w <= h:
        new_w = target_short
        new_h = int(h * (target_short / w))
    else:
        new_h = target_short
        new_w = int(w * (target_short / h))
    return img.resize((new_w, new_h), Image.LANCZOS)


def process_image(src_path: Path, dst_path: Path, method: str):
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            if method == "resize_crop":
                # follow common ImageNet preprocess: resize shorter side to 256 then center-crop 224
                img = resize_shorter_side(img, 256)
                img = center_crop(img, 224)
            else:
                # direct resize to 224x224
                img = img.resize((224, 224), Image.LANCZOS)

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst_path, format="JPEG", quality=95)
            return True, src_path
    except Exception as e:
        return False, (src_path, str(e))


def gather_image_paths(root: Path):
    # include common image extensions
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=False, default="data/Dataset/PetImages",
                   help="Input dataset root (class subfolders expected)")
    p.add_argument("--output", "-o", required=False, default="data/processed/PetImages_224",
                   help="Output root where processed images are saved")
    p.add_argument("--method", choices=["resize_crop", "resize"], default="resize_crop",
                   help="Resize method: 'resize_crop' (shorter->256 then center-crop 224) or 'resize' (direct 224x224)")
    p.add_argument("--workers", "-w", type=int, default=4, help="Parallel workers")
    p.add_argument("--skip-existing", action="store_true", help="Skip images that already exist in output")
    args = p.parse_args()

    src_root = Path(args.input)
    dst_root = Path(args.output)

    if not src_root.exists():
        raise SystemExit(f"Input path not found: {src_root}")

    paths = list(gather_image_paths(src_root))
    if not paths:
        raise SystemExit(f"No images found under: {src_root}")

    tasks = []
    for src in paths:
        # Preserve relative structure
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        # force jpg extension
        dst = dst.with_suffix('.jpg')
        if args.skip_existing and dst.exists():
            continue
        tasks.append((src, dst))

    good = 0
    bad = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_image, s, d, args.method) for s, d in tasks]
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), unit="img"):
            ok, info = fut.result()
            if ok:
                good += 1
            else:
                bad.append(info)

    print(f"Processed: {good} images; Failed: {len(bad)}")
    if bad:
        print("Some failures (first 10):")
        for b in bad[:10]:
            print(b)


if __name__ == "__main__":
    main()
