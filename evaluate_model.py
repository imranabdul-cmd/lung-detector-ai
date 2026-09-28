import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_images"
MODEL_PATH = BASE_DIR / "model" / "lung_cancer_model.keras"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def main() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError("dataset_images/ was not found.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model/lung_cancer_model.keras was not found. Run train_model.py first.")

    model = keras.models.load_model(MODEL_PATH)

    validation_dataset = keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_names = validation_dataset.class_names
    y_true = []
    y_pred = []

    for images, labels in validation_dataset:
        processed = keras.applications.efficientnet.preprocess_input(images)
        probabilities = model.predict(processed, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(probabilities, axis=1).tolist())

    labels_index = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_index)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_index,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "sample_count": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "classes": class_names,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    with (ARTIFACTS_DIR / "evaluation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Validation Confusion Matrix",
    )

    threshold = cm.max() / 2 if cm.size else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(json.dumps({
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "samples": metrics["sample_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
