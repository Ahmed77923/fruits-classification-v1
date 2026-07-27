# trainer.py
from typing import List

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fruits.notebook_info.nnote import EpochMetrics


# ===================================================================
# Training
# ===================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochMetrics:
    """
    Train the model for one epoch.

    Returns:
        EpochMetrics: Average loss and accuracy.
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return EpochMetrics(
        loss=epoch_loss,
        accuracy=epoch_accuracy,
    )


# ===================================================================
# Validation
# ===================================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    """
    Evaluate the model for one epoch.

    Returns:
        EpochMetrics: Average loss and accuracy.
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return EpochMetrics(loss=epoch_loss,accuracy=epoch_accuracy,)


# ===================================================================
# Fit
# ===================================================================

def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    scheduler=None,
):
    """
    Train the model for multiple epochs.

    Returns:
        tuple[list[EpochMetrics], list[EpochMetrics]]
    """

    train_history: List[EpochMetrics] = []
    val_history: List[EpochMetrics] = []

    best_accuracy = 0.0
    start_time = time.time()
    for epoch in range(1, epochs + 1):

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

        train_history.append(train_metrics)
        val_history.append(val_metrics)

        
        # Save the best model
        if val_metrics.accuracy > best_accuracy:
            best_accuracy = val_metrics.accuracy

            torch.save(
                model.state_dict(),
                "checkpoints/best_model.pth"
            )

            print(
                f"Best model saved | "
                f"Val Accuracy: {best_accuracy:.4f}"
            )
        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {train_metrics.loss:.4f} | "
            f"Train Acc: {train_metrics.accuracy:.4f} | "
            f"Val Loss: {val_metrics.loss:.4f} | "
            f"Val Acc: {val_metrics.accuracy:.4f}"
        )
    end_time = time.time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(
        f"\nTraining completed in "
        f"{minutes}m {seconds}s"
)

    return train_history, val_history


