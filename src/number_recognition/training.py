"""Training and evaluation loops for image classifiers."""

from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import Optimizer


def run_epoch(
    model: nn.Module,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    loss_function: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None = None,
) -> dict[str, float]:
    """Run one complete training pass, or one gradient-free evaluation pass."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for batch_images, batch_labels in data_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits = model(batch_images)
            loss = loss_function(logits, batch_labels)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = batch_labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Cannot run an epoch over an empty data loader")

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_model(
    model: nn.Module,
    train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    validation_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: Optimizer,
    epochs: int,
    device: torch.device,
    loss_function: nn.Module | None = None,
) -> list[dict[str, float | int]]:
    """Train for a fixed number of epochs and return train/validation metrics."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    if loss_function is None:
        loss_function = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, loss_function, device, optimizer=optimizer
        )
        validation_metrics = run_epoch(
            model, validation_loader, loss_function, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": validation_metrics["loss"],
                "val_accuracy": validation_metrics["accuracy"],
            }
        )
        print(
            f"Époque {epoch}/{epochs} | "
            f"train loss={train_metrics['loss']:.4f}, "
            f"acc={train_metrics['accuracy']:.3f} | "
            f"val loss={validation_metrics['loss']:.4f}, "
            f"acc={validation_metrics['accuracy']:.3f}"
        )

    return history
