"""Shape classifier demo package."""

from .data import (
    CLASS_NAMES,
    COLORS,
    SHAPES,
    batch_features,
    generate_synthetic_dataset,
    generate_synthetic_image,
    load_cifar10_subset,
)
from .system import (
    ShapeClassifierSystem,
    CLIPClassifierSystem,  # Backwards compatibility alias
    ProjectionWeights,
    ClassifierWeights,
    ColorWeights,
    ColorMLPWeights,
    ShapeWeights,
    accuracy_metric,
    classify,
    classify_color,
    classify_shape,
    combine_predictions,
    extract_color_features,
    extract_shape_features,
    project,
)

__all__ = [
    # Data
    "CLASS_NAMES",
    "COLORS",
    "SHAPES",
    "batch_features",
    "generate_synthetic_dataset",
    "generate_synthetic_image",
    "load_cifar10_subset",
    # System
    "ShapeClassifierSystem",
    "CLIPClassifierSystem",
    "ProjectionWeights",
    "ClassifierWeights",
    "ColorWeights",
    "ColorMLPWeights",
    "ShapeWeights",
    "classify_color",
    "classify_shape",
    "combine_predictions",
    "extract_color_features",
    "extract_shape_features",
    "accuracy_metric",
    # Legacy aliases
    "project",
    "classify",
]
