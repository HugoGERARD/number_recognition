"""HTTP API for digit prediction from uploaded images."""

from io import BytesIO
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
import torch
from torch import nn

from .inference import predict_image

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_DIMENSION = 4096


class HealthResponse(BaseModel):
    status: str
    device: str


class PredictionResponse(BaseModel):
    digit: int
    confidence: float


def create_app(model: nn.Module, device: torch.device) -> FastAPI:
    """Create the API around an already-loaded model."""
    app = FastAPI(title="Number Recognition API", version="0.1.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", device=str(device))

    @app.post("/predict", response_model=PredictionResponse)
    def predict(file: Annotated[UploadFile, File(description="PNG ou JPEG d'un chiffre")]) -> PredictionResponse:
        content = file.file.read(MAX_IMAGE_BYTES + 1)
        if not content:
            raise HTTPException(status_code=400, detail="Le fichier image est vide.")
        if len(content) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image trop volumineuse (maximum 5 Mo).")

        try:
            with Image.open(BytesIO(content)) as image:
                if image.width > MAX_IMAGE_DIMENSION or image.height > MAX_IMAGE_DIMENSION:
                    raise HTTPException(
                        status_code=413,
                        detail="Image trop grande (maximum 4096 × 4096 pixels).",
                    )
                image.load()
                digit, confidence = predict_image(model, image, device)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise HTTPException(status_code=400, detail="Le fichier n'est pas une image valide.") from error

        return PredictionResponse(digit=digit, confidence=confidence)

    return app
