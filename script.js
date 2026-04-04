const fileInput = document.querySelector("#file-input");
const fileName = document.querySelector("#file-name");
const uploadPreview = document.querySelector("#upload-preview");
const predictButton = document.querySelector("#predict-button");
const dropzone = document.querySelector("#dropzone");
const resultsSection = document.querySelector("#results");
const resultImage = document.querySelector("#result-image");
const predictionLabel = document.querySelector("#prediction-label");
const predictionConfidence = document.querySelector("#prediction-confidence");
const meaningText = document.querySelector("#meaning-text");
const predictionBox = document.querySelector("#prediction-box");
const scoreList = document.querySelector("#score-list");
const exampleButtons = document.querySelectorAll("[data-example]");
const contactForm = document.querySelector("#contact-form");
const statusBanner = document.querySelector("#status-banner");
const backendNote = document.querySelector("#backend-note");

const chartLines = {
  trainAccuracy: document.querySelector("#train-accuracy-line"),
  valAccuracy: document.querySelector("#val-accuracy-line"),
  trainLoss: document.querySelector("#train-loss-line"),
  valLoss: document.querySelector("#val-loss-line"),
};

const scoreClassNames = {
  Adenocarcinoma: "score-adeno",
  Normal: "score-normal",
  Squamous: "score-squamous",
};

const demoResults = {
  normal: {
    label: "Normal",
    confidence: 99.68,
    image: "assets/lung-detector/result-normal.jpg",
    meaning:
      "The tissue appears normal, with no signs of cancerous cells detected by the model.",
    scores: [
      { name: "Adenocarcinoma", value: 1.2 },
      { name: "Normal", value: 99.68 },
      { name: "Squamous", value: 0.8 },
    ],
  },
  adeno: {
    label: "Adenocarcinoma",
    confidence: 92.44,
    image: "assets/lung-detector/result-adeno.jpg",
    meaning:
      "This example simulates adenocarcinoma-like features from the reference project until your trained model is added.",
    scores: [
      { name: "Adenocarcinoma", value: 92.44 },
      { name: "Normal", value: 5.31 },
      { name: "Squamous", value: 2.25 },
    ],
  },
  squamous: {
    label: "Squamous",
    confidence: 88.24,
    image: "assets/lung-detector/example-squamous.jpg",
    meaning:
      "This example simulates squamous cell carcinoma-like features until your trained model is added.",
    scores: [
      { name: "Adenocarcinoma", value: 7.45 },
      { name: "Normal", value: 4.31 },
      { name: "Squamous", value: 88.24 },
    ],
  },
};

function formatPercent(value) {
  return `${Number(value).toFixed(2)}%`;
}

function setBackendNote(message, state = "") {
  backendNote.textContent = message;
  backendNote.classList.remove("is-error", "is-success");
  if (state) {
    backendNote.classList.add(state);
  }
}

function getPredictionTone(label) {
  if (label === "Normal") {
    return {
      bg: "#eef3ff",
      border: "#3463e8",
      text: "#234fcf",
    };
  }

  if (label === "Squamous") {
    return {
      bg: "#fff7db",
      border: "#efc94c",
      text: "#936b00",
    };
  }

  if (label === "Adenocarcinoma") {
    return {
      bg: "#fff0f5",
      border: "#e76c99",
      text: "#be3e6d",
    };
  }

  return {
    bg: "#f5f7fb",
    border: "#b5bfd3",
    text: "#55607a",
  };
}

function buildScoreRows(scores) {
  scoreList.innerHTML = scores
    .map((score) => {
      const className = scoreClassNames[score.name] || "score-normal";
      return `
        <div class="score-row">
          <span>${score.name}</span>
          <div class="score-track">
            <div class="score-fill ${className}" style="width: ${score.value}%"></div>
          </div>
          <strong>${formatPercent(score.value)}</strong>
        </div>
      `;
    })
    .join("");
}

function renderResult(result) {
  predictionLabel.textContent = result.label;
  predictionConfidence.textContent = `Confidence: ${formatPercent(result.confidence)}`;
  meaningText.textContent = result.meaning;

  if (result.image) {
    resultImage.src = result.image;
  }

  const tone = getPredictionTone(result.label);
  predictionBox.style.background = tone.bg;
  predictionBox.style.borderLeftColor = tone.border;
  predictionLabel.style.color = tone.text;

  buildScoreRows(result.scores);
  resultsSection.classList.remove("hidden");
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function buildPolyline(values, { invert = false } = {}) {
  if (!Array.isArray(values) || values.length < 2) {
    return null;
  }

  const xStart = 40;
  const xEnd = 390;
  const yTop = 24;
  const yBottom = 190;
  const maxValue = Math.max(...values);
  const minValue = Math.min(...values);
  const range = maxValue - minValue || 1;

  return values
    .map((value, index) => {
      const x = xStart + (index * (xEnd - xStart)) / (values.length - 1);
      const normalized = invert
        ? (value - minValue) / range
        : 1 - (value - minValue) / range;
      const y = yTop + normalized * (yBottom - yTop);
      return `${x},${y}`;
    })
    .join(" ");
}

function updateCharts(metrics) {
  if (!metrics || !Array.isArray(metrics.train_accuracy) || metrics.train_accuracy.length < 2) {
    return;
  }

  const accuracyTrain = buildPolyline(metrics.train_accuracy);
  const accuracyVal = buildPolyline(metrics.val_accuracy);
  const lossTrain = buildPolyline(metrics.train_loss, { invert: true });
  const lossVal = buildPolyline(metrics.val_loss, { invert: true });

  if (accuracyTrain) {
    chartLines.trainAccuracy.setAttribute("points", accuracyTrain);
  }
  if (accuracyVal) {
    chartLines.valAccuracy.setAttribute("points", accuracyVal);
  }
  if (lossTrain) {
    chartLines.trainLoss.setAttribute("points", lossTrain);
  }
  if (lossVal) {
    chartLines.valLoss.setAttribute("points", lossVal);
  }
}

function showPreview(file) {
  fileName.textContent = file.name;
  const reader = new FileReader();
  reader.onload = (event) => {
    uploadPreview.src = event.target.result;
    uploadPreview.hidden = false;
  };
  reader.readAsDataURL(file);
}

async function fetchBackendStatus() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error("Health check failed");
    }

    const payload = await response.json();
    if (payload.service.ready) {
      setBackendNote(
        `Model loaded successfully from ${payload.service.model_path}.`,
        "is-success"
      );
    } else {
      setBackendNote(
        "No trained model found yet. Add your histopathology dataset, run train_model.py, then predictions will become live."
      );
    }
  } catch (error) {
    setBackendNote(
      "Frontend is ready, but the Flask backend is not running yet. Start app.py to enable real prediction.",
      "is-error"
    );
  }
}

async function fetchTrainingMetrics() {
  try {
    const response = await fetch("/metrics");
    if (!response.ok) {
      return;
    }

    const payload = await response.json();
    updateCharts(payload);
  } catch (error) {
    // Keep the default chart lines when the backend is unavailable.
  }
}

fileInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (!file) {
    return;
  }

  showPreview(file);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
  });
});

dropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (!file) {
    return;
  }

  fileInput.files = event.dataTransfer.files;
  showPreview(file);
});

predictButton.addEventListener("click", async () => {
  const selectedFile = fileInput.files[0];

  if (!selectedFile) {
    alert("Please select a histopathology image first.");
    return;
  }

  predictButton.disabled = true;
  predictButton.textContent = "Predicting...";

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Prediction failed.");
    }

    renderResult({
      ...payload,
      image: uploadPreview.src,
    });
    setBackendNote("Prediction completed using the Python backend.", "is-success");
  } catch (error) {
    setBackendNote(error.message, "is-error");
    alert(error.message);
  } finally {
    predictButton.disabled = false;
    predictButton.textContent = "Predict";
  }
});

exampleButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const exampleKey = button.dataset.example;
    renderResult(demoResults[exampleKey]);
  });
});

contactForm.addEventListener("submit", (event) => {
  event.preventDefault();
  statusBanner.classList.remove("hidden");
  statusBanner.scrollIntoView({ behavior: "smooth", block: "center" });
  contactForm.reset();
});

fetchBackendStatus();
fetchTrainingMetrics();
