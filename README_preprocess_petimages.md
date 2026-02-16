# PetImages Preprocessing

This repository includes a small script to preprocess images under `data/Dataset/PetImages` to 224x224 RGB images suitable for standard CNNs.

Files added:
- `preprocess_petimages.py` — main preprocessing script
- `requirements-preprocess.txt` — minimal Python deps

Quick run:

```bash
python preprocess_petimages.py \
  --input data/Dataset/PetImages \
  --output data/processed/PetImages_224 \
  --method resize_crop \
  --workers 8
```

Notes:
- Default `resize_crop` replicates ImageNet preprocessing: shorter side -> 256, then center-crop 224.
- `resize` directly resizes to 224x224.
- The script preserves class subfolders and saves images as JPEGs.
