"""Shape classifier HyperSystem implementation."""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple
from PIL import Image
from hyperfunc import (
    HyperSystem,
    LoRAWeight,
    hyperfunction,
)


# Shape classifier: circularity (1) -> 2 shapes (simple linear)
ShapeWeights = LoRAWeight.create(out_dim=2, in_dim=1, noise_rank=1)


# Color MLP: RGB (3) -> hidden (8) -> 5 colors
# We store both weight matrices as a single tensor for vmap compatibility
HIDDEN_DIM = 8
COLOR_IN = 3
COLOR_OUT = 5


@dataclass
class ColorMLPWeights:
    """MLP weights for color classifier: 3 -> 8 -> 5.

    Stores W1 (8x3) and W2 (5x8) as a single flattened tensor for vmap.
    """
    weight: torch.Tensor  # Flattened: (8*3 + 5*8,) = (64,)

    @classmethod
    def shape(cls) -> Tuple[int, ...]:
        return (HIDDEN_DIM * COLOR_IN + COLOR_OUT * HIDDEN_DIM,)

    @classmethod
    def dim(cls) -> int:
        return HIDDEN_DIM * COLOR_IN + COLOR_OUT * HIDDEN_DIM  # 24 + 40 = 64

    @classmethod
    def noise_rank(cls) -> int:
        return 4  # Low-rank noise for the flattened weight vector

    def to_tensor(self, device=None, dtype=torch.float32) -> torch.Tensor:
        t = self.weight.to(dtype=dtype)
        if device is not None:
            t = t.to(device)
        return t

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "ColorMLPWeights":
        return cls(weight=t.view(-1))

    def get_w1_w2(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract W1 (8x3) and W2 (5x8) from flattened weight."""
        w1_size = HIDDEN_DIM * COLOR_IN
        w1 = self.weight[:w1_size].view(HIDDEN_DIM, COLOR_IN)
        w2 = self.weight[w1_size:].view(COLOR_OUT, HIDDEN_DIM)
        return w1, w2


# Legacy alias for backwards compatibility
ColorWeights = ColorMLPWeights


def extract_color_features(image) -> torch.Tensor:
    """Extract color features from image.

    Computes mean RGB of non-white pixels (the shape).
    Returns 3-dim vector normalized to [0, 1].
    """
    if isinstance(image, Image.Image):
        img = image.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
    else:
        arr = np.array(image).astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

    # Find non-white pixels (the shape) - white is (1, 1, 1)
    is_shape = np.any(arr < 0.95, axis=2)

    if is_shape.sum() > 0:
        shape_pixels = arr[is_shape]
        mean_rgb = shape_pixels.mean(axis=0)
    else:
        mean_rgb = arr.mean(axis=(0, 1))

    return torch.tensor(mean_rgb, dtype=torch.float32)


def extract_shape_features(image) -> torch.Tensor:
    """Extract shape features using circularity.

    Circularity = 4 * pi * area / perimeter^2
    - Circle: ~1.0
    - Square: ~0.785 (pi/4)

    Returns 1-dim vector.
    """
    if isinstance(image, Image.Image):
        img = image.convert("L")  # Grayscale
        arr = np.array(img).astype(np.float32) / 255.0
    else:
        arr = np.array(image).astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if len(arr.shape) == 3:
            arr = arr.mean(axis=2)

    # Binary mask of shape (non-white pixels)
    mask = arr < 0.95

    # Area = number of shape pixels
    area = mask.sum()

    if area == 0:
        return torch.tensor([0.5], dtype=torch.float32)

    # Perimeter = number of edge pixels
    # Edge pixel = shape pixel with at least one non-shape neighbor
    padded = np.pad(mask, 1, mode='constant', constant_values=False)

    # Check 4-connectivity neighbors
    neighbors = (
        padded[:-2, 1:-1].astype(int) +  # top
        padded[2:, 1:-1].astype(int) +   # bottom
        padded[1:-1, :-2].astype(int) +  # left
        padded[1:-1, 2:].astype(int)     # right
    )

    # Edge pixels are shape pixels where not all 4 neighbors are shape
    edge_mask = mask & (neighbors < 4)
    perimeter = edge_mask.sum()

    if perimeter == 0:
        return torch.tensor([0.5], dtype=torch.float32)

    # Circularity formula
    circularity = 4 * np.pi * area / (perimeter ** 2)

    # Clamp to reasonable range and center around 1.0
    # Circle ~1.26, Square ~0.80, centered: Circle ~0.26, Square ~-0.20
    circularity = np.clip(circularity, 0, 1.5) - 1.0

    return torch.tensor([circularity], dtype=torch.float32)


@hyperfunction(hp_type=ColorMLPWeights, optimize_hparams=True)
async def classify_color(features: torch.Tensor, hp) -> torch.Tensor:
    """Classify color with MLP: RGB (3) -> hidden (8) -> 5 colors."""
    w1, w2 = hp.get_w1_w2()
    hidden = torch.relu(features @ w1.T)  # (3,) @ (3, 8).T -> (8,)
    logits = hidden @ w2.T                 # (8,) @ (8, 5).T -> (5,)
    return torch.softmax(logits, dim=-1)


@hyperfunction(hp_type=ShapeWeights, optimize_hparams=True)
async def classify_shape(features: torch.Tensor, hp) -> torch.Tensor:
    """Classify shape: circularity (1) -> 2 shapes."""
    logits = features @ hp.weight.T
    return torch.softmax(logits, dim=-1)


def combine_predictions(color_probs: torch.Tensor, shape_probs: torch.Tensor) -> torch.Tensor:
    """Combine color and shape predictions into 10-class probabilities.

    Classes are ordered: color_idx * 2 + shape_idx
    e.g., red_circle=0, red_square=1, blue_circle=2, blue_square=3, ...
    """
    # Outer product: P(color_i, shape_j) = P(color_i) * P(shape_j)
    # Result shape: (5, 2) -> flatten to (10,)
    combined = torch.outer(color_probs, shape_probs)
    return combined.flatten()


class ShapeClassifierSystem(HyperSystem):
    """Simple shape + color classifier with factorized architecture.

    This system demonstrates:
    1. Factorized classification: separate color and shape classifiers
    2. Two independent trainable knobs (color weights, shape weights)
    3. Deterministic combination of predictions

    Architecture:
        Image -> color_features (3) -> classify_color (3->5) -> color_probs
              -> shape_features (1) -> classify_shape (1->2) -> shape_probs
              -> combine_predictions -> 10-class probs
    """

    def __init__(
        self,
        class_names: list[str],
        **kwargs,
    ):
        """Initialize the classifier system.

        Args:
            class_names: List of class names (must have 10 classes)
            **kwargs: Additional arguments passed to HyperSystem
        """
        super().__init__(**kwargs)
        if len(class_names) != 10:
            raise ValueError(f"Expected 10 classes, got {len(class_names)}")
        self.class_names = class_names

    async def run(
        self,
        color_features: torch.Tensor,
        shape_features: torch.Tensor,
    ) -> dict:
        """Classify using pre-extracted features.

        Args:
            color_features: Pre-extracted color features (3,) tensor
            shape_features: Pre-extracted shape features (1,) tensor

        Returns:
            Dict with class prediction and confidence
        """
        # Classify independently (two trainable knobs)
        color_probs = await classify_color(color_features)   # 3 -> 5
        shape_probs = await classify_shape(shape_features)   # 1 -> 2

        # Combine predictions (deterministic)
        probs = combine_predictions(color_probs, shape_probs)  # 10

        # Get prediction
        pred_idx = probs.argmax().item()
        pred_class = self.class_names[pred_idx]
        confidence = probs[pred_idx].item()

        return {
            "class": pred_class,
            "class_idx": pred_idx,
            "confidence": confidence,
            "probs": probs,
            "color_probs": color_probs,
            "shape_probs": shape_probs,
        }


def accuracy_metric(preds: list, expected: list[dict]) -> float:
    """Classification accuracy metric (higher is better)."""
    correct = 0
    for pred, exp in zip(preds, expected):
        if isinstance(pred, dict):
            pred_idx = pred["class_idx"]
        elif isinstance(pred, torch.Tensor):
            pred_idx = pred.argmax().item()
        else:
            pred_idx = pred

        if pred_idx == exp["class_idx"]:
            correct += 1
    return correct / len(preds)


# Backwards compatibility
CLIPClassifierSystem = ShapeClassifierSystem
ClassifierWeights = ColorWeights  # Legacy alias
ProjectionWeights = ColorWeights  # Legacy alias
classify = classify_color  # Legacy alias
project = classify_color  # Legacy alias
