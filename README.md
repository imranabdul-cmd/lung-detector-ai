# Lung Detector AI

This project now includes a real Python backend for histopathology image prediction and a TensorFlow training script for your lung cancer dataset.

## Project Flow

1. Put your dataset inside `dataset_images/`
2. Train the model with `python train_model.py`
3. Start the app with `python app.py`
4. Open `http://127.0.0.1:5000`

## Expected Dataset Structure

```text
dataset_images/
  Adenocarcinoma/
    image1.jpg
    image2.jpg
  Normal/
    image1.jpg
    image2.jpg
  Squamous/
    image1.jpg
    image2.jpg
```

## Files

- `app.py`: Flask server and `/predict` API
- `train_model.py`: model training and artifact export
- `artifacts/class_names.json`: generated after training
- `artifacts/model_config.json`: generated after training
- `artifacts/training_metrics.json`: generated after training
- `model/lung_cancer_model.keras`: generated trained model

## Notes

- The frontend calls the Python backend directly.
- If the model is missing, the page will tell you to train or add the model first.
- The training charts on the page update from `artifacts/training_metrics.json` after training.
