import time
import io
import os
import json
import base64
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Any

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

# Configuration
IMAGE_SIZE = (224, 224)
DEFAULT_MODEL_PATH = Path("baseline_cnn_best.h5")
LOG_FILE = "inference.log"

# Logging setup
logger = logging.getLogger("inference_service")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)
fh = logging.FileHandler(LOG_FILE)
fh.setFormatter(fmt)
logger.addHandler(fh)

# Simple in-memory metrics
metrics = {
    "request_count": 0,
    "total_latency_ms": 0.0,
}


def load_model(model_uri: Optional[str] = None) -> Any:
    """Load a Keras model either from MLflow run URI (e.g. 'runs:/<id>/model')
    or from a local filepath (HDF5/TF SavedModel).

    This function performs lazy imports of `mlflow` and `tensorflow` so the
    module can be imported in lightweight test runs that don't require the
    heavy ML libraries.
    """
    # Lazy import mlflow + mlflow.keras
    mlflow = None
    mlflow_keras = None
    try:
        import mlflow as _mlflow
        import mlflow.keras as _mlflow_keras
        mlflow = _mlflow
        mlflow_keras = _mlflow_keras
    except Exception:
        mlflow = None
        mlflow_keras = None

    # Lazy import tensorflow only when needed
    tf = None
    try:
        import tensorflow as _tf
        tf = _tf
    except Exception:
        tf = None

    if model_uri and mlflow_keras is not None:
        logger.info(f"Loading model from URI: {model_uri}")
        try:
            model = mlflow_keras.load_model(model_uri)
            logger.info("Loaded model via MLflow")
            return model
        except Exception as e:
            logger.warning(f"mlflow.keras.load_model failed: {e}; trying local path fallback")

    # fallback: local path using tensorflow if available
    if tf is not None and DEFAULT_MODEL_PATH.exists():
        logger.info(f"Loading local model: {DEFAULT_MODEL_PATH}")
        return tf.keras.models.load_model(str(DEFAULT_MODEL_PATH))

    raise FileNotFoundError("No model found: provide --model-uri or place baseline_cnn_best.h5 in working dir")


def preprocess_pil_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img).astype("float32") / 255.0
    return arr


def decode_base64_image(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data))


app = FastAPI(title="PetImages Inference Service")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    req_id = str(uuid.uuid4())
    t0 = time.time()
    metrics["request_count"] += 1
    try:
        response = await call_next(request)
        return response
    finally:
        latency_ms = (time.time() - t0) * 1000.0
        metrics["total_latency_ms"] += latency_ms
        logger.info(f"req_id={req_id} method={request.method} path={request.url.path} latency_ms={latency_ms:.1f}")


MODEL = None
LABELS: Optional[List[str]] = None


@app.on_event("startup")
def startup_event():
    global MODEL, LABELS
    # MODEL_URI can be set via env var; otherwise default local
    model_uri = os.environ.get("MODEL_URI")
    try:
        MODEL = load_model(model_uri)
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        MODEL = None
    # Optional labels file
    labels_path = os.environ.get("LABELS_JSON")
    if labels_path and Path(labels_path).exists():
        try:
            LABELS = json.loads(Path(labels_path).read_text())
            logger.info(f"Loaded {len(LABELS)} labels from {labels_path}")
        except Exception as e:
            logger.warning(f"Failed to load labels file {labels_path}: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/metrics")
def get_metrics():
    rc = metrics.get("request_count", 0)
    total = metrics.get("total_latency_ms", 0.0)
    avg = (total / rc) if rc else 0.0
    return {"request_count": rc, "avg_latency_ms": avg}


@app.post("/predict")
async def predict(files: Optional[List[UploadFile]] = File(None), payload: Optional[dict] = None):
    """Prediction endpoint.
    Accepts either multipart file upload(s) (form field `files`) or JSON body with `instances` list of base64-encoded images.
    Returns per-instance probabilities and predicted class index (and label if available).
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    images = []
    # prefer uploaded files
    if files:
        for f in files:
            try:
                img = Image.open(io.BytesIO(await f.read()))
                images.append(preprocess_pil_image(img))
            except Exception as e:
                logger.exception(f"Failed to read uploaded file {f.filename}: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid image {f.filename}")
    else:
        # accept JSON with base64 instances
        try:
            body = await Request.json if isinstance(payload, Request) else payload
        except Exception:
            body = payload
        if not body:
            raise HTTPException(status_code=400, detail="No input provided")
        instances = body.get("instances") if isinstance(body, dict) else None
        if not instances:
            raise HTTPException(status_code=400, detail="JSON body must contain 'instances' list of base64 images")
        for b in instances:
            try:
                img = decode_base64_image(b)
                images.append(preprocess_pil_image(img))
            except Exception as e:
                logger.exception(f"Failed to decode base64 image: {e}")
                raise HTTPException(status_code=400, detail="Invalid base64 image in instances")

    batch = np.stack(images, axis=0)
    # ensure batch shape matches model input
    try:
        preds = MODEL.predict(batch)
    except Exception as e:
        logger.exception(f"Model prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed")

    results = []
    for p in preds:
        class_idx = int(np.argmax(p))
        label = LABELS[class_idx] if LABELS and class_idx < len(LABELS) else None
        results.append({"probabilities": p.tolist(), "predicted_class": class_idx, "predicted_label": label})

    # log request/response summary
    logger.info(f"predict num_instances={len(results)}")
    return JSONResponse({"predictions": results})


def run():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost", help="Host to bind the server to")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model-uri", type=str, default=None, help="MLflow model URI (runs:/<id>/model) or path to model file")
    p.add_argument("--labels", type=str, default=None, help="Optional JSON file with list of label names")
    args = p.parse_args()

    if args.model_uri:
        os.environ["MODEL_URI"] = args.model_uri
    if args.labels:
        os.environ["LABELS_JSON"] = args.labels

    # Start uvicorn
    uvicorn.run("inference_service:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    run()
