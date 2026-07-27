import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fruits.config.config import Config
from fruits.dataset.dataset import FruitsDataset
from fruits.models.cnn import CNNModel
from fruits.engine.trainer import fit


def main():

    train_dataset = FruitsDataset(
        root_dir=Config.TRAIN_PATH,
        mode="train",
    )

    val_dataset = FruitsDataset(
        root_dir=Config.TEST_PATH,
        mode="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model = CNNModel(
        num_classes=len(train_dataset.class_names)
    ).to(Config.DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
    )

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=Config.DEVICE,
        epochs=Config.EPOCHS,
    )


if __name__ == "__main__":
    main()