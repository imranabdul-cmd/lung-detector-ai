import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from tensorflow import keras


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
UPLOADS_DIR = BASE_DIR / "uploads"

CLASS_NAMES = ["Adenocarcinoma", "Normal", "Squamous"]
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_MODEL_CANDIDATES = (
    MODEL_DIR / "lung_cancer_model.keras",
    MODEL_DIR / "lung_cancer_model.h5",
    MODEL_DIR / "saved_model",
)


app = Flask(__name__, static_folder=".", static_url_path="")
UPLOADS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)


class PredictionService:
    def __init__(self) -> None:
        self.model = None
        self.model_path = None
        self.class_names = self._load_class_names()
        self.image_size = self._load_image_size()
        self.metrics = self._load_metrics()
        self._load_model()

    def _load_class_names(self) -> List[str]:
        class_map_path = ARTIFACTS_DIR / "class_names.json"
        if class_map_path.exists():
            with class_map_path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if isinstance(value, list) and value:
                return value
        return CLASS_NAMES

    def _load_image_size(self) -> Tuple[int, int]:
        settings_path = ARTIFACTS_DIR / "model_config.json"
        if settings_path.exists():
            with settings_path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            image_size = value.get("image_size")
            if isinstance(image_size, list) and len(image_size) == 2:
                return int(image_size[0]), int(image_size[1])
        return DEFAULT_IMAGE_SIZE

    def _load_metrics(self) -> Dict:
        metrics_path = ARTIFACTS_DIR / "training_metrics.json"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return {
            "status": "missing",
            "message": "Training metrics are not available yet. Run train_model.py after adding your dataset.",
        }

    def _load_model(self) -> None:
        for candidate in DEFAULT_MODEL_CANDIDATES:
            if candidate.exists():
                self.model = keras.models.load_model(candidate)
                self.model_path = str(candidate)
                return

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize(self.image_size)
        array = keras.utils.img_to_array(image).astype("float32")
        array = keras.applications.efficientnet.preprocess_input(array)
        return np.expand_dims(array, axis=0)

    def predict(self, image_bytes: bytes) -> Dict:
        if not self.model:
            raise FileNotFoundError(
                "No trained model found. Add a trained model to the model folder or run train_model.py first."
            )

        batch = self.preprocess(image_bytes)
        scores = self.model.predict(batch, verbose=0)[0]

        if scores.ndim != 1:
            scores = np.ravel(scores)

        # Convert logits to probabilities if needed.
        if np.max(scores) > 1.0 or np.min(scores) < 0.0:
            scores = keras.activations.softmax(scores).numpy()

        label_index = int(np.argmax(scores))
        label = self.class_names[label_index]
        confidence = float(scores[label_index]) * 100

        explanation = {
            "Adenocarcinoma": "The model found features consistent with adenocarcinoma patterns in this histopathology image.",
            "Normal": "The model found no dominant malignant pattern and classified the tissue as normal.",
            "Squamous": "The model found features consistent with squamous cell carcinoma patterns in this histopathology image.",
        }.get(label, "Prediction generated successfully.")

        return {
            "label": label,
            "confidence": confidence,
            "meaning": explanation,
            "scores": [
                {
                    "name": class_name,
                    "value": float(scores[index]) * 100,
                }
                for index, class_name in enumerate(self.class_names)
            ],
        }


service = PredictionService()


def serialize_status() -> Dict:
    return {
        "ready": service.is_ready,
        "model_path": service.model_path,
        "classes": service.class_names,
        "image_size": list(service.image_size),
        "metrics": service.metrics,
    }


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": serialize_status()})


@app.route("/metrics")
def metrics():
    return jsonify(service.metrics)


@app.route("/predict", methods=["POST"])
def predict():
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Please upload a histopathology image file."}), 400

    try:
        image_bytes = uploaded_file.read()
        result = service.predict(image_bytes)
        result["service"] = serialize_status()
        return jsonify(result)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc), "service": serialize_status()}), 503
    except Exception as exc:  # pragma: no cover - defensive API guard
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
