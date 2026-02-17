#!/usr/bin/env bash
set -e

echo "Entry point: check MODEL_URI and download model if set"
if [ -n "${MODEL_URI-}" ]; then
  echo "MODEL_URI is set: $MODEL_URI"
  python download_model.py || echo "download_model.py failed"
else
  echo "MODEL_URI not set; will use local baseline_cnn_best.h5 if present"
fi

echo "Starting inference service"
exec python inference_service.py --host 0.0.0.0 --port 8000
