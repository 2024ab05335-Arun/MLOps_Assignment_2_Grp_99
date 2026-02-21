#!/usr/bin/env python3
"""Collect a small batch of requests against the inference endpoint and save responses with true labels.

Sends images from a test directory (assumes subdirectories are class labels, e.g. test/Cat, test/Dog),
posts each image to `/predict` as multipart form `files`, and records the response JSON.

Output: CSV with columns: image_path,true_label,status_code,predicted_label,response_json
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm


def guess_pred(response_json):
    # Try a few common keys to extract a single predicted label
    if not isinstance(response_json, dict):
        return None
    for key in ("predicted_label", "prediction", "label", "predictions", "result"):
        if key in response_json:
            val = response_json[key]
            if isinstance(val, list) and val:
                return val[0]
            return val
    # fallback: look for first string value
    for v in response_json.values():
        if isinstance(v, str):
            return v
        if isinstance(v, list) and v and isinstance(v[0], str):
            return v[0]
    return None


def send_image(url, image_path):
    files = {"files": (os.path.basename(image_path), open(image_path, "rb"))}
    try:
        r = requests.post(f"{url.rstrip('/')}/predict", files=files, timeout=20)
    except Exception as e:
        return None, None, str(e)
    try:
        j = r.json()
    except Exception:
        j = {"text": r.text}
    pred = guess_pred(j)
    return r.status_code, pred, j


def main():
    parser = argparse.ArgumentParser(description="Collect requests and true labels from test images")
    parser.add_argument("--url", default="http://localhost:8000", help="Inference service base URL")
    parser.add_argument("--image-dir", required=True, help="Directory containing class subfolders with images")
    parser.add_argument("--output", default="monitoring_results.csv", help="CSV output path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images per class (0 = all)")
    parser.add_argument("--skip-unknown", action="store_true", help="Skip files that cannot be opened")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Image directory not found: {image_dir}")
        sys.exit(2)

    rows = []
    classes = [p for p in sorted(image_dir.iterdir()) if p.is_dir()]
    for cls in classes:
        files = [p for p in sorted(cls.iterdir()) if p.is_file()]
        if args.limit and args.limit > 0:
            files = files[: args.limit]
        for fp in tqdm(files, desc=f"Scanning {cls.name}"):
            try:
                status, pred, resp = send_image(args.url, str(fp))
            except Exception as e:
                if args.skip_unknown:
                    continue
                status = None
                pred = None
                resp = {"error": str(e)}
            rows.append({
                "image_path": str(fp),
                "true_label": cls.name,
                "status_code": status,
                "predicted_label": pred,
                "response_json": json.dumps(resp, ensure_ascii=False),
            })

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "true_label", "status_code", "predicted_label", "response_json"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # quick summary
    total = len(rows)
    matched = sum(1 for r in rows if r["predicted_label"] is not None and str(r["predicted_label"]).lower() == str(r["true_label"]).lower())
    print(f"Wrote {outp} with {total} rows. Simple accuracy (exact match): {matched}/{total} = {matched/total if total else 0:.3f}")


if __name__ == "__main__":
    main()
