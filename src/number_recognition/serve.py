"""Local development entry point for the prediction API."""

from pathlib import Path

import torch
import uvicorn

from .api import create_app
from .model import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "digit_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model checkpoint not found at {MODEL_PATH}. "
        "Run the training notebook and its save cell first."
    )

model = load_model(MODEL_PATH, DEVICE)
app = create_app(model, DEVICE)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
