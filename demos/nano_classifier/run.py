#!/usr/bin/env python3
"""
Nano Classifier - A minimal ES training demo without HyperSystem.

This demo uses TorchEggrollES directly to train a simple classifier,
bypassing the HyperSystem tracing infrastructure. This helps verify
that the ES optimizer itself works correctly.

Architecture:
- Color classifier: 3 (RGB) -> 5 colors (linear layer)
- Shape classifier: 1 (circularity) -> 2 shapes (linear layer)
- Combined: outer product -> 10 classes

This is the same task as shape_classifier demo, but without HyperSystem.
If this works but shape_classifier doesn't, the bug is in evaluate_population.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict, Any

from hyperfunc.es import TorchEggrollES


# ============================================================
# Data Generation (same as shape_classifier)
# ============================================================

COLORS = ["red", "blue", "green", "yellow", "cyan"]
SHAPES = ["circle", "square"]
CLASS_NAMES = [f"{c}_{s}" for c in COLORS for s in SHAPES]

COLOR_MAP = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
}


def generate_shape_image(color: str, shape: str, size: int = 32) -> Image.Image:
    """Generate a synthetic image of a colored shape."""
    img = Image.new("RGB", (size, size), (128, 128, 128))
    draw = ImageDraw.Draw(img)
    rgb = COLOR_MAP[color]
    margin = size // 8

    if shape == "circle":
        draw.ellipse([margin, margin, size - margin, size - margin], fill=rgb)
    else:
        draw.rectangle([margin, margin, size - margin, size - margin], fill=rgb)

    return img


def extract_features(img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features from image.

    Returns:
        color_features: RGB mean (3,) normalized to [0, 1]
        shape_features: aspect ratio based feature (1,) - circles ~0.5, squares ~-0.5
    """
    arr = np.array(img)

    # Color: mean RGB
    color = arr.mean(axis=(0, 1)) / 255.0
    color_features = torch.tensor(color, dtype=torch.float32)

    # Shape: use contour moments for better circularity
    # Threshold to get binary mask (shape vs background)
    # Use max absolute difference across all RGB channels (not grayscale) to detect any color
    bg_rgb = np.array([128, 128, 128])
    diff = np.abs(arr.astype(np.float32) - bg_rgb).max(axis=2)  # Max diff across channels
    mask = diff > 50  # Threshold

    # Calculate moments
    area = mask.sum()
    if area == 0:
        shape_features = torch.tensor([0.0], dtype=torch.float32)
        return color_features, shape_features

    # Find bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    height = row_max - row_min + 1
    width = col_max - col_min + 1

    # Compactness: area / bounding_box_area
    # Circles have compactness ~0.785 (pi/4), squares have compactness ~1.0
    bb_area = height * width
    compactness = area / bb_area if bb_area > 0 else 0

    # Transform: circles -> positive, squares -> negative
    # Compactness: circle ~0.785, square ~1.0
    # Map to: circle -> +0.5, square -> -0.5
    shape_feature = (0.9 - compactness) * 4.0  # circle: (0.9-0.785)*4 = 0.46, square: (0.9-1.0)*4 = -0.4
    shape_features = torch.tensor([shape_feature], dtype=torch.float32)

    return color_features, shape_features


def generate_dataset(n_per_class: int = 20, seed: int = 42, verbose: bool = False) -> List[Dict[str, Any]]:
    """Generate synthetic dataset."""
    np.random.seed(seed)
    data = []

    for color_idx, color in enumerate(COLORS):
        for shape_idx, shape in enumerate(SHAPES):
            class_idx = color_idx * 2 + shape_idx
            class_name = CLASS_NAMES[class_idx]

            shape_feats_list = []
            for _ in range(n_per_class):
                img = generate_shape_image(color, shape)
                color_feat, shape_feat = extract_features(img)
                shape_feats_list.append(shape_feat.item())

                data.append({
                    "color_features": color_feat,
                    "shape_features": shape_feat,
                    "class_idx": class_idx,
                    "class_name": class_name,
                })

            if verbose:
                mean_shape = np.mean(shape_feats_list)
                print(f"  {class_name}: shape_feat mean = {mean_shape:.4f}")

    np.random.shuffle(data)
    return data


# ============================================================
# Model (simple nn.Module without HyperSystem)
# ============================================================

class FactorizedClassifier(nn.Module):
    """
    Factorized classifier using two MLPs + outer product.

    Each sub-classifier is a 2-layer MLP with ReLU activation.
    This gives the model more capacity to learn non-linear boundaries.
    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        # Color MLP: 3 -> hidden -> 5 (RGB to 5 colors)
        self.color_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),
        )
        # Shape MLP: 1 -> hidden -> 2 (circularity to circle/square)
        self.shape_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        # Xavier-like init for all layers
        for layer in [self.color_mlp[0], self.color_mlp[2], self.shape_mlp[0], self.shape_mlp[2]]:
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, color_features: torch.Tensor, shape_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            color_features: (batch, 3) RGB features
            shape_features: (batch, 1) circularity features

        Returns:
            logits: (batch, 10) class logits
        """
        # Get probabilities from each classifier
        color_probs = F.softmax(self.color_mlp(color_features), dim=-1)  # (batch, 5)
        shape_probs = F.softmax(self.shape_mlp(shape_features), dim=-1)  # (batch, 2)

        # Outer product for each batch element
        # combined[i, j] = color_probs[i] * shape_probs[j]
        # For batch processing: (batch, 5, 1) * (batch, 1, 2) -> (batch, 5, 2)
        combined = color_probs.unsqueeze(-1) * shape_probs.unsqueeze(-2)

        # Flatten to (batch, 10)
        logits = combined.view(color_features.shape[0], -1)

        return logits


def evaluate_model(model: nn.Module, data: List[Dict[str, Any]]) -> float:
    """Evaluate model accuracy on dataset."""
    model.eval()
    correct = 0
    total = len(data)

    with torch.no_grad():
        for ex in data:
            color_feat = ex["color_features"].unsqueeze(0)
            shape_feat = ex["shape_features"].unsqueeze(0)

            logits = model(color_feat, shape_feat)
            pred = logits.argmax(dim=-1).item()

            if pred == ex["class_idx"]:
                correct += 1

    return correct / total


# ============================================================
# Main Training Loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Nano Classifier - ES without HyperSystem")
    parser.add_argument("--steps", type=int, default=50, help="ES optimization steps")
    parser.add_argument("--pop-size", type=int, default=32, help="Population size")
    parser.add_argument("--sigma", type=float, default=0.1, help="Noise scale")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--rank", type=int, default=2, help="Low-rank noise rank")
    parser.add_argument("--n-per-class", type=int, default=20, help="Examples per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Generate data
    print("Generating dataset...")
    train_data = generate_dataset(n_per_class=args.n_per_class, seed=args.seed, verbose=True)
    print(f"Generated {len(train_data)} examples across {len(CLASS_NAMES)} classes")
    print(f"Classes: {CLASS_NAMES}")

    # Create model
    print("\nCreating FactorizedClassifier (2-layer MLPs)...")
    model = FactorizedClassifier(hidden_dim=16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Color MLP: 3 -> 16 -> 5")
    print(f"Shape MLP: 1 -> 16 -> 2")
    print(f"Total parameters: {n_params}")

    # Initial accuracy
    initial_acc = evaluate_model(model, train_data)
    print(f"\nInitial accuracy: {initial_acc:.2%}")

    # Create ES optimizer
    print(f"\nCreating TorchEggrollES (pop_size={args.pop_size}, sigma={args.sigma}, lr={args.lr}, rank={args.rank})...")
    es = TorchEggrollES(
        model=model,
        pop_size=args.pop_size,
        sigma=args.sigma,
        lr=args.lr,
        rank=args.rank,
        antithetic=True,
        normalize_fitness=True,
    )

    # Training loop
    print(f"\nOptimizing ({args.steps} steps)...")
    print("-" * 50)

    for step in range(args.steps):
        # Define fitness function for this step
        def fitness_fn(m: nn.Module) -> float:
            return evaluate_model(m, train_data)

        # ES step
        mean_fitness = es.step(fitness_fn)

        # Log progress
        if step % 5 == 0 or step == args.steps - 1:
            current_acc = evaluate_model(model, train_data)
            print(f"Step {step:3d}: mean_fitness={mean_fitness:.4f}, accuracy={current_acc:.2%}")

    print("-" * 50)

    # Final accuracy
    final_acc = evaluate_model(model, train_data)
    print(f"\nFinal accuracy: {final_acc:.2%}")
    print(f"Improvement: {final_acc - initial_acc:+.2%}")

    # Show learned weights summary
    print("\nModel learned successfully via ES optimization.")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    class_correct = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}

    with torch.no_grad():
        for ex in train_data:
            color_feat = ex["color_features"].unsqueeze(0)
            shape_feat = ex["shape_features"].unsqueeze(0)

            logits = model(color_feat, shape_feat)
            pred = logits.argmax(dim=-1).item()
            expected = ex["class_idx"]

            class_total[expected] += 1
            if pred == expected:
                class_correct[expected] += 1

    for i, name in enumerate(CLASS_NAMES):
        total = class_total[i]
        correct = class_correct[i]
        acc = correct / total if total > 0 else 0
        print(f"  {name:15}: {acc:.1%} ({correct}/{total})")


if __name__ == "__main__":
    main()
