import os
import sys
from pathlib import Path

MODEL_PATH = Path("baseline_cnn_best.h5")


def download_model_from_mlflow(uri: str):
    try:
        import mlflow.keras
    except Exception as e:
        print("mlflow not available, cannot download model:", e, file=sys.stderr)
        return False

    try:
        print(f"Downloading model from MLflow URI: {uri}")
        model = mlflow.keras.load_model(uri)
        # Save in a standard Keras HDF5 file so inference_service can load it
        model.save(str(MODEL_PATH))
        print(f"Saved model to {MODEL_PATH}")
        return True
    except Exception as e:
        print("Failed to download or save model:", e, file=sys.stderr)
        return False


def main():
    uri = os.environ.get("MODEL_URI")
    if not uri:
        print("No MODEL_URI provided; skipping model download")
        return

    ok = download_model_from_mlflow(uri)
    if not ok:
        print("Model download failed; container will still start but inference may fail")


if __name__ == '__main__':
    main()
