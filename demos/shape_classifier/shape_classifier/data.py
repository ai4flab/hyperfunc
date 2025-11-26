"""Data generation and loading for CLIP classifier demo."""

import random
import torch
from PIL import Image, ImageDraw
from hyperfunc import Example

# Synthetic dataset: colored shapes
# Using colors that are linearly separable in RGB space
COLORS = ["red", "blue", "green", "yellow", "cyan"]
SHAPES = ["circle", "square"]
CLASS_NAMES = [f"{c}_{s}" for c in COLORS for s in SHAPES]  # 10 classes

COLOR_MAP = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
}


def generate_synthetic_image(
    class_name: str,
    size: int = 224,
    add_noise: bool = True,
    seed: int | None = None,
) -> Image.Image:
    """Generate a synthetic image for a given class.

    Creates a simple image with a colored shape on a white background.
    Optionally adds random position/size variation for diversity.

    Args:
        class_name: Class name in format "color_shape" (e.g., "red_circle")
        size: Image size in pixels (square)
        add_noise: Whether to add random position/size variation
        seed: Random seed for reproducibility

    Returns:
        PIL Image with the colored shape
    """
    if seed is not None:
        random.seed(seed)

    color, shape = class_name.split("_")
    rgb = COLOR_MAP[color]

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Base margin
    base_margin = size // 4

    if add_noise:
        # Add random variation to position and size
        margin_var = size // 16
        margin = base_margin + random.randint(-margin_var, margin_var)
        offset_x = random.randint(-margin_var, margin_var)
        offset_y = random.randint(-margin_var, margin_var)
    else:
        margin = base_margin
        offset_x = offset_y = 0

    bbox = [
        margin + offset_x,
        margin + offset_y,
        size - margin + offset_x,
        size - margin + offset_y,
    ]

    if shape == "circle":
        draw.ellipse(bbox, fill=rgb)
    else:  # square
        draw.rectangle(bbox, fill=rgb)

    return img


def generate_synthetic_dataset(
    n_per_class: int = 20,
    size: int = 224,
    add_noise: bool = True,
    seed: int = 42,
) -> list[Example]:
    """Generate synthetic training examples with pre-extracted features.

    Args:
        n_per_class: Number of examples per class
        size: Image size in pixels
        add_noise: Whether to add random variation
        seed: Random seed for reproducibility

    Returns:
        List of Example objects with pre-extracted color/shape features
    """
    from .system import extract_color_features, extract_shape_features

    random.seed(seed)
    examples = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        for i in range(n_per_class):
            img = generate_synthetic_image(
                class_name,
                size=size,
                add_noise=add_noise,
                seed=seed + class_idx * 1000 + i if seed else None,
            )

            # Pre-extract features for fast training
            color_features = extract_color_features(img)
            shape_features = extract_shape_features(img)
            examples.append(Example(
                {
                    "color_features": color_features,
                    "shape_features": shape_features,
                },
                {"class_idx": class_idx, "class_name": class_name}
            ))

    # Shuffle examples
    random.shuffle(examples)
    return examples


def batch_features(examples: list[Example]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack pre-extracted features into batched tensors.

    Args:
        examples: List of Examples with color_features and shape_features

    Returns:
        Tuple of (color_features_batch, shape_features_batch)
        - color_features_batch: (num_examples, 3)
        - shape_features_batch: (num_examples, 1)
    """
    color_features = torch.stack([ex.inputs["color_features"] for ex in examples])
    shape_features = torch.stack([ex.inputs["shape_features"] for ex in examples])
    return color_features, shape_features


def load_cifar10_subset(
    n_per_class: int = 50,
    split: str = "train",
) -> list[Example]:
    """Load a subset of CIFAR-10 for real data testing.

    Args:
        n_per_class: Number of examples per class to load
        split: Dataset split ("train" or "test")

    Returns:
        List of Example objects with CIFAR-10 images and labels
    """
    from datasets import load_dataset

    ds = load_dataset("cifar10", split=split)
    cifar_classes = ds.features["label"].names

    examples = []
    counts = {i: 0 for i in range(10)}

    for item in ds:
        label = item["label"]
        if counts[label] < n_per_class:
            examples.append(Example(
                {"image": item["img"]},
                {"class_idx": label, "class_name": cifar_classes[label]}
            ))
            counts[label] += 1
        if all(c >= n_per_class for c in counts.values()):
            break

    return examples
