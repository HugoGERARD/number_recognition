"""CNN architecture and checkpoint helpers for MNIST digit classification."""

from pathlib import Path

import torch
from torch import nn


class DigitCNN(nn.Module):
    """Two convolutional blocks followed by a dense 10-class classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_maps = self.features(images)
        flattened_features = self.flatten(feature_maps)
        return self.classifier(flattened_features)


def save_model(model: DigitCNN, path: str | Path) -> Path:
    """Save model weights and return the checkpoint path."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def load_model(path: str | Path, device: torch.device) -> DigitCNN:
    """Load a DigitCNN checkpoint onto the requested device."""
    model = DigitCNN().to(device)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model
