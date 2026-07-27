from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

from fruits.config.config import Config
from fruits.dataset.dataset import FruitsDataset
from fruits.models.cnn import CNNModel


class PredictorError(Exception):
    """Raised when the predictor cannot be initialized or used safely."""


@dataclass(frozen=True)
class PredictionItem:
    class_name: str
    confidence: float


class FruitPredictor:
    """Inference helper for loading the trained model and predicting fruit classes."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        class_names: list[str],
        device: Optional[torch.device] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or Config.DEVICE
        self.class_names = class_names

        self._validate_inputs()
        self.transform = FruitsDataset.default_val_transform()
        self.model = self._load_model()

    def _validate_inputs(self) -> None:
        if not self.class_names:
            raise PredictorError("Class mapping is empty. Cannot run inference.")

        if not self.checkpoint_path.is_file():
            raise PredictorError(
                f"Checkpoint not found at: {self.checkpoint_path}"
            )

    def _load_model(self) -> CNNModel:
        model = CNNModel(num_classes=len(self.class_names)).to(self.device)

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        except Exception as exc:
            raise PredictorError(f"Failed to load checkpoint: {exc}") from exc

        state_dict: dict[str, Any]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise PredictorError("Checkpoint format is not supported for this model.")

        try:
            model.load_state_dict(state_dict)
        except Exception as exc:
            raise PredictorError(
                "Checkpoint is incompatible with CNNModel architecture. "
                f"Details: {exc}"
            ) from exc

        model.eval()
        return model

    @staticmethod
    def derive_class_names_from_training_dir(train_path: str | Path) -> list[str]:
        """Derive class names using the same deterministic logic as FruitsDataset."""
        root = Path(train_path)
        if not root.is_dir():
            raise PredictorError(
                f"Training data directory does not exist: {root}"
            )

        class_names = sorted(
            child.name for child in root.iterdir() if child.is_dir()
        )

        if not class_names:
            raise PredictorError(
                f"No class folders found in training directory: {root}"
            )

        return class_names

    def predict(self, image: Image.Image, top_k: int = 3) -> dict[str, Any]:
        """Predict class probabilities for a single PIL image."""
        if not isinstance(image, Image.Image):
            raise PredictorError("Input must be a PIL.Image.Image instance.")

        if top_k <= 0:
            raise PredictorError("top_k must be greater than zero.")

        if top_k > len(self.class_names):
            top_k = len(self.class_names)

        rgb_image = image.convert("RGB")

        try:
            input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        except Exception as exc:
            raise PredictorError(f"Image preprocessing failed: {exc}") from exc

        with torch.inference_mode():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)

        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

        top_predictions: list[PredictionItem] = []
        for prob, idx in zip(top_probs[0].tolist(), top_indices[0].tolist()):
            top_predictions.append(
                PredictionItem(
                    class_name=self.class_names[idx],
                    confidence=float(prob),
                )
            )

        best = top_predictions[0]

        return {
            "class_name": best.class_name,
            "confidence": best.confidence,
            "top_predictions": [
                {
                    "class_name": item.class_name,
                    "confidence": item.confidence,
                }
                for item in top_predictions
            ],
        }
