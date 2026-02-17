"""Client examples for the inference service.

Usage examples:

1) curl - multipart/form-data upload (Linux/macOS):

   curl -X POST "http://localhost:8000/predict" -F "files=@/path/to/image.jpg"

   PowerShell (Windows):
   curl -X POST "http://localhost:8000/predict" -F "files=@C:\\path\\to\\image.jpg"

2) curl - JSON with base64 image:

   B64=$(base64 -w 0 image.jpg) && \
   curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"instances\":[\"$B64\"]}"

This script provides two client helpers and a small CLI.
"""
import argparse
import base64
import json
import os
from pathlib import Path

import requests


def predict_file(url: str, image_path: str):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    files = {"files": (image_path.name, open(image_path, "rb"), "image/jpeg")}
    resp = requests.post(f"{url.rstrip('/')}/predict", files=files)
    resp.raise_for_status()
    return resp.json()


def predict_base64(url: str, image_path: str):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    b = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    payload = {"instances": [b]}
    resp = requests.post(f"{url.rstrip('/')}/predict", json=payload)
    resp.raise_for_status()
    return resp.json()


def main():
    p = argparse.ArgumentParser(description="Simple client for the PetImages inference service")
    p.add_argument("--url", default="http://localhost:8000", help="Base URL of the inference service")
    p.add_argument("--image", required=True, help="Path to image file to send")
    p.add_argument("--mode", choices=["file", "base64"], default="file", help="Send as multipart file or base64 JSON")
    args = p.parse_args()

    if args.mode == "file":
        out = predict_file(args.url, args.image)
    else:
        out = predict_base64(args.url, args.image)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
