const API_URL = "http://127.0.0.1:8000";
const canvas = document.querySelector("#drawing-canvas");
const context = canvas.getContext("2d");
const clearButton = document.querySelector("#clear-button");
const predictButton = document.querySelector("#predict-button");
const canvasHint = document.querySelector("#canvas-hint");
const result = document.querySelector("#result");
const resultDigit = document.querySelector("#result-digit");
const resultMessage = document.querySelector("#result-message");
const errorMessage = document.querySelector("#error-message");
const apiStatus = document.querySelector("#api-status");
const apiStatusText = document.querySelector("#api-status-text");

let drawing = false;
let hasStroke = false;

context.lineWidth = 20;
context.lineCap = "round";
context.lineJoin = "round";
context.strokeStyle = "#111";

function paintCanvasWhite() {
  context.fillStyle = "#fff";
  context.fillRect(0, 0, canvas.width, canvas.height);
}

paintCanvasWhite();

function pointFromEvent(event) {
  const bounds = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - bounds.left) * (canvas.width / bounds.width),
    y: (event.clientY - bounds.top) * (canvas.height / bounds.height),
  };
}

canvas.addEventListener("pointerdown", (event) => {
  drawing = true;
  hasStroke = true;
  canvas.setPointerCapture(event.pointerId);
  const point = pointFromEvent(event);
  context.beginPath();
  context.moveTo(point.x, point.y);
  context.lineTo(point.x + 0.1, point.y + 0.1);
  context.stroke();
  canvasHint.hidden = true;
  predictButton.disabled = false;
  result.hidden = true;
  errorMessage.hidden = true;
});

canvas.addEventListener("pointermove", (event) => {
  if (!drawing) return;
  const point = pointFromEvent(event);
  context.lineTo(point.x, point.y);
  context.stroke();
});

function stopDrawing() {
  drawing = false;
  context.beginPath();
}

canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);

clearButton.addEventListener("click", () => {
  context.clearRect(0, 0, canvas.width, canvas.height);
  paintCanvasWhite();
  context.beginPath();
  drawing = false;
  hasStroke = false;
  canvasHint.hidden = false;
  predictButton.disabled = true;
  result.hidden = true;
  errorMessage.hidden = true;
});

function canvasToPng() {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Impossible de convertir le dessin en PNG."));
    }, "image/png");
  });
}

predictButton.addEventListener("click", async () => {
  if (!hasStroke) return;

  predictButton.disabled = true;
  predictButton.textContent = "Prédiction…";
  errorMessage.hidden = true;

  try {
    const image = await canvasToPng();
    const formData = new FormData();
    formData.append("file", image, "drawing.png");

    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: formData,
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || `Erreur HTTP ${response.status}`);
    }

    resultDigit.textContent = body.digit;
    resultMessage.textContent = `Confiance : ${(body.confidence * 100).toFixed(1)} %`;
    result.hidden = false;
  } catch (error) {
    errorMessage.textContent = `Prédiction impossible : ${error.message}`;
    errorMessage.hidden = false;
  } finally {
    predictButton.disabled = !hasStroke;
    predictButton.innerHTML = 'Prédire <span aria-hidden="true">→</span>';
  }
});

async function checkApi() {
  try {
    const response = await fetch(`${API_URL}/health`);
    if (!response.ok) throw new Error("API indisponible");
    apiStatus.dataset.state = "online";
    apiStatusText.textContent = "API connectée";
  } catch {
    apiStatus.dataset.state = "offline";
    apiStatusText.textContent = "API hors ligne";
  }
}

checkApi();
