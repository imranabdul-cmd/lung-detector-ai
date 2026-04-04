import json
import shutil
import tempfile
from pathlib import Path

import tensorflow as tf
from tensorflow import keras


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_images"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_DIR = BASE_DIR / "model"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
HEAD_EPOCHS = 6
FINETUNE_EPOCHS = 10
SEED = 42


def build_model(num_classes: int) -> tuple[keras.Model, keras.Model]:
    augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.08),
            keras.layers.RandomZoom(0.12),
            keras.layers.RandomContrast(0.1),
            keras.layers.RandomTranslation(0.08, 0.08),
        ],
        name="augmentation",
    )

    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
        weights="imagenet",
    )
    backbone.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = keras.applications.efficientnet.preprocess_input(x)
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.35)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, backbone


def ensure_dataset() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            "Dataset folder not found. Create a 'dataset_images' folder with class subfolders such as "
            "'Adenocarcinoma', 'Normal', and 'Squamous'."
        )


def should_stage_dataset(dataset_dir: Path) -> bool:
    return "OneDrive" in str(dataset_dir)


def prepare_training_dataset() -> Path:
    if not should_stage_dataset(DATASET_DIR):
        return DATASET_DIR

    temp_root = Path(tempfile.gettempdir()) / "lung_detector_ai_dataset"
    if temp_root.exists():
        shutil.rmtree(temp_root)

    print(f"Copying dataset to local temp folder for training: {temp_root}")
    shutil.copytree(DATASET_DIR, temp_root)
    return temp_root


def dataset_from_directory(dataset_dir: Path, subset: str):
    return keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset=subset,
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )


def save_artifacts(history: keras.callbacks.History, class_names):
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    history_data = history.history
    metrics_payload = {
        "epochs": list(range(1, len(history_data["accuracy"]) + 1)),
        "train_accuracy": history_data["accuracy"],
        "val_accuracy": history_data["val_accuracy"],
        "train_loss": history_data["loss"],
        "val_loss": history_data["val_loss"],
    }

    with (ARTIFACTS_DIR / "training_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    with (ARTIFACTS_DIR / "class_names.json").open("w", encoding="utf-8") as handle:
        json.dump(list(class_names), handle, indent=2)

    with (ARTIFACTS_DIR / "model_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"image_size": list(IMAGE_SIZE)}, handle, indent=2)


def merge_histories(*histories: keras.callbacks.History) -> dict:
    merged = {
        "accuracy": [],
        "val_accuracy": [],
        "loss": [],
        "val_loss": [],
    }
    for history in histories:
        for key in merged:
            merged[key].extend(history.history.get(key, []))
    return merged


def make_callbacks(checkpoint_path: Path):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
    ]


def unfreeze_backbone(backbone: keras.Model) -> None:
    backbone.trainable = True
    for layer in backbone.layers[:-40]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False


def main():
    ensure_dataset()
    training_dataset_dir = prepare_training_dataset()

    train_dataset = dataset_from_directory(training_dataset_dir, "training")
    val_dataset = dataset_from_directory(training_dataset_dir, "validation")

    class_names = train_dataset.class_names
    autotune = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(autotune)
    val_dataset = val_dataset.prefetch(autotune)

    MODEL_DIR.mkdir(exist_ok=True)
    checkpoint_path = MODEL_DIR / "lung_cancer_best.keras"

    model, backbone = build_model(num_classes=len(class_names))
    callbacks = make_callbacks(checkpoint_path)

    print("Stage 1: training classification head")
    head_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=HEAD_EPOCHS,
        callbacks=callbacks,
    )

    print("Stage 2: fine-tuning upper backbone layers")
    unfreeze_backbone(backbone)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    fine_tune_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=HEAD_EPOCHS + FINETUNE_EPOCHS,
        initial_epoch=len(head_history.history["accuracy"]),
        callbacks=callbacks,
    )

    if checkpoint_path.exists():
        model = keras.models.load_model(checkpoint_path)

    model.save(MODEL_DIR / "lung_cancer_model.keras")
    combined_history = merge_histories(head_history, fine_tune_history)
    history_wrapper = type("HistoryWrapper", (), {"history": combined_history})()
    save_artifacts(history_wrapper, class_names)

    print("Training finished successfully.")
    print(f"Model saved to: {MODEL_DIR / 'lung_cancer_model.keras'}")
    print(f"Classes: {class_names}")


if __name__ == "__main__":
    main()
