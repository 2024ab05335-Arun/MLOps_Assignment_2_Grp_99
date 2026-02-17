"""
mlflow_workflow.py

Command-line script to run the same MLflow experiment as the notebook.
- Configures MLflow to use a local `mlruns` folder
- Loads train/val/test datasets from a prepared directory
- Trains a simple CNN, logs params/metrics/artifacts to MLflow
- Evaluates on test set and logs confusion matrix and loss/accuracy plot

Usage:
    python mlflow_workflow.py --data-dir ./data/processed/PetImages_224_split --epochs 5

Requires: tensorflow, mlflow, scikit-learn, matplotlib, numpy
"""
from pathlib import Path
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import mlflow
import mlflow.keras
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from mlflow.tracking import MlflowClient


# Explicit MLflow per-epoch metrics callback
class MLflowMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            try:
                mlflow.log_metric(k, float(v), step=epoch)
            except Exception as e:
                print(f"Failed to log metric {k}: {e}")


def setup_mlflow(base_dir: Path, experiment_name: str):
    mlruns_dir = (base_dir / "mlruns").resolve()
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    # Windows-friendly file URI
    mlflow.set_tracking_uri(f"file:///{mlruns_dir.as_posix()}")
    mlflow.set_experiment(experiment_name)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    return mlruns_dir


def prepare_datasets(data_dir: Path, image_size=(224, 224), batch_size=32, seed=42):
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'
    for d in (train_dir, val_dir, test_dir):
        if not d.exists():
            raise FileNotFoundError(f"Required directory not found: {d}")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        shuffle=True,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        shuffle=False,
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        shuffle=False,
    )

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
    ])
    normalization_layer = layers.Rescaling(1.0 / 255.0)
    AUTOTUNE = tf.data.AUTOTUNE

    def prepare(ds, augment=False):
        ds = ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
        if augment:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
            return ds.shuffle(1000).prefetch(AUTOTUNE)
        return ds.prefetch(AUTOTUNE)

    classes = train_ds.class_names
    train_ds = prepare(train_ds, augment=True)
    val_ds = prepare(val_ds, augment=False)
    test_ds = prepare(test_ds, augment=False)

    return train_ds, val_ds, test_ds, classes


def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def plot_and_save_history(history, out_path: Path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history.get('loss', []), label='train_loss')
    ax[0].plot(history.history.get('val_loss', []), label='val_loss')
    ax[0].set_title('Loss')
    ax[0].legend()

    acc_hist = history.history.get('accuracy', history.history.get('acc', []))
    ax[1].plot(acc_hist, label='train_acc')
    ax[1].plot(history.history.get('val_accuracy', history.history.get('val_acc', [])), label='val_acc')
    ax[1].set_title('Accuracy')
    ax[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path))
    plt.close(fig)


def train_and_log(train_ds, val_ds, test_ds, classes, base_dir: Path, epochs=3, batch_size=32):
    input_shape = (224, 224, 3)
    num_classes = len(classes)
    model = build_model(input_shape, num_classes)

    # Start MLflow run
    with mlflow.start_run(run_name='baseline_cnn') as run:
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('image_size', input_shape[:2])
        mlflow.log_param('num_classes', num_classes)

        checkpoint_file = base_dir / 'baseline_cnn_best.h5'
        mc = callbacks.ModelCheckpoint(str(checkpoint_file), monitor='val_accuracy', save_best_only=True, verbose=1)
        mlflow_cb = MLflowMetricsCallback()
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[mc, mlflow_cb])

        # Save and log training curves
        lossacc_path = base_dir / 'loss_accuracy.png'
        plot_and_save_history(history, lossacc_path)
        try:
            mlflow.log_artifact(str(lossacc_path))
        except Exception as e:
            print('Failed to log loss/acc artifact:', e)

        # Log best model
        best = tf.keras.models.load_model(str(checkpoint_file))
        mlflow.keras.log_model(best, 'model')

        run_id = run.info.run_id
        print('Training finished. Run id:', run_id)

    return run_id


def evaluate_and_log(test_ds, classes, base_dir: Path, run_id: str):
    # Load model from run
    model_uri = f"runs:/{run_id}/model"
    print('Loading model from', model_uri)
    loaded = mlflow.keras.load_model(model_uri)

    # Collect true labels and predicted probabilities
    y_true_list = []
    for x_batch, y_batch in test_ds:
        y_true_list.append(y_batch.numpy())
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred_proba = loaded.predict(test_ds)
    y_true_cls = np.argmax(y_true, axis=1)
    y_pred_cls = np.argmax(y_pred_proba, axis=1)

    acc = accuracy_score(y_true_cls, y_pred_cls)
    print(f'Test accuracy: {acc:.4f}')

    # Confusion matrix
    cm = confusion_matrix(y_true_cls, y_pred_cls)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    cm_path = base_dir / 'confusion_matrix.png'
    plt.tight_layout()
    fig.savefig(str(cm_path))
    plt.close(fig)

    # Log evaluation as nested run
    with mlflow.start_run(run_name='evaluation', nested=True) as eval_run:
        mlflow.log_metric('test_accuracy', float(acc))
        try:
            mlflow.log_artifact(str(cm_path))
            lossacc_path = base_dir / 'loss_accuracy.png'
            if lossacc_path.exists():
                mlflow.log_artifact(str(lossacc_path))
        except Exception as e:
            print('Failed to log artifacts:', e)
        mlflow.keras.log_model(loaded, 'evaluated_model')

    print('Evaluation run id:', eval_run.info.run_id)


def inspect_runs(base_dir: Path, experiment_name: str, top_n: int = 5):
    """List recent runs for the experiment, print params/metrics and download common artifacts."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print('Experiment not found:', experiment_name)
        return
    runs = client.search_runs(exp.experiment_id, order_by=['attributes.start_time DESC'], max_results=top_n)
    if not runs:
        print('No runs found for experiment', experiment_name)
        return
    print(f'Found {len(runs)} runs for {experiment_name}')
    for r in runs:
        print('---')
        print('Run id:', r.info.run_id, 'status:', r.info.status)
        print('Params:', r.data.params)
        print('Metrics:', r.data.metrics)
        try:
            arts = client.list_artifacts(r.info.run_id, path='')
            print('Artifacts:', [a.path for a in arts])
        except Exception as e:
            print('Failed to list artifacts for run', r.info.run_id, e)

    # Download common artifacts from most recent run
    latest = runs[0]
    run_id = latest.info.run_id
    out_dir = base_dir / 'mlruns_inspect' / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print('Inspecting latest run:', run_id)
    candidates = ['loss_accuracy.png', 'confusion_matrix.png']
    try:
        available = [a.path for a in client.list_artifacts(run_id, path='')]
    except Exception:
        available = []
    for name in candidates:
        if name in available:
            print('Downloading', name)
            local_path = client.download_artifacts(run_id, name, dst_path=str(out_dir))
            print('Saved to', local_path)
        else:
            for a in available:
                if a.endswith(name) or name in a:
                    print('Downloading nested', a)
                    local_path = client.download_artifacts(run_id, a, dst_path=str(out_dir))
                    print('Saved to', local_path)
                    break
    print('Artifacts downloaded to', out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, default='data/processed/PetImages_224_split')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--image-size', type=int, nargs=2, default=[224, 224])
    p.add_argument('--experiment-name', type=str, default='petimages_baseline_workflow')
    p.add_argument('--inspect', action='store_true', help='Inspect recent runs and download artifacts after run')
    args = p.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = (base_dir / args.data_dir) if not Path(args.data_dir).is_absolute() else Path(args.data_dir)

    # Setup mlflow
    setup_mlflow(base_dir, args.experiment_name)

    train_ds, val_ds, test_ds, classes = prepare_datasets(data_dir, image_size=tuple(args.image_size), batch_size=args.batch_size)

    run_id = train_and_log(train_ds, val_ds, test_ds, classes, base_dir, epochs=args.epochs, batch_size=args.batch_size)

    evaluate_and_log(test_ds, classes, base_dir, run_id)
    if args.inspect:
        inspect_runs(base_dir, args.experiment_name)


if __name__ == '__main__':
    main()
