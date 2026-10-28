"""Image preprocessing and model prediction shared by interfaces."""

from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import transforms

_IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ]
)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert a handwritten digit to the 1x28x28 MNIST input convention."""
    grayscale_image = image.convert("L")
    image_tensor = _IMAGE_TRANSFORM(grayscale_image)
    if image_tensor.mean().item() > 0.5:
        image_tensor = 1.0 - image_tensor
    return image_tensor


def predict_image(
    model: nn.Module, image: Image.Image, device: torch.device
) -> tuple[int, float]:
    """Return the predicted digit and its softmax confidence."""
    image_batch = preprocess_image(image).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(image_batch), dim=1)
        confidence, predicted_digit = probabilities.max(dim=1)
    return predicted_digit.item(), confidence.item()
