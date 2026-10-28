import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from number_recognition.inference import predict_image, preprocess_image
from number_recognition.model import DigitCNN, load_model, save_model
from number_recognition.training import run_epoch


class ConstantClassifier(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros((images.size(0), 10), device=images.device)


class NumberRecognitionTests(unittest.TestCase):
    def test_cnn_returns_ten_class_logits(self) -> None:
        model = DigitCNN()
        logits = model(torch.zeros(2, 1, 28, 28))
        self.assertEqual(tuple(logits.shape), (2, 10))

    def test_run_epoch_processes_batches_with_equal_loss(self) -> None:
        dataset = TensorDataset(
            torch.zeros(2, 1, 28, 28), torch.tensor([0, 1])
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        metrics = run_epoch(
            ConstantClassifier(), loader, nn.CrossEntropyLoss(), torch.device("cpu")
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["loss"], 2.302585, places=5)

    def test_preprocess_drawing_matches_mnist_shape_and_contrast(self) -> None:
        image = Image.new("L", (280, 280), color=255)
        ImageDraw.Draw(image).line((140, 50, 140, 230), fill=0, width=20)
        tensor = preprocess_image(image)
        self.assertEqual(tuple(tensor.shape), (1, 28, 28))
        self.assertEqual(tensor[0, 0, 0].item(), 0.0)
        self.assertGreater(tensor.max().item(), 0.5)

    def test_predict_image_returns_digit_and_confidence(self) -> None:
        image = Image.new("L", (280, 280), color=255)
        ImageDraw.Draw(image).line((140, 50, 140, 230), fill=0, width=20)
        digit, confidence = predict_image(
            ConstantClassifier(), image, torch.device("cpu")
        )
        self.assertEqual(digit, 0)
        self.assertAlmostEqual(confidence, 0.1, places=6)

    def test_model_checkpoint_round_trip(self) -> None:
        model = DigitCNN()
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "digit_cnn.pth"
            save_model(model, checkpoint_path)
            restored_model = load_model(checkpoint_path, torch.device("cpu"))
        self.assertEqual(
            model(torch.zeros(1, 1, 28, 28)).shape,
            restored_model(torch.zeros(1, 1, 28, 28)).shape,
        )


if __name__ == "__main__":
    unittest.main()
