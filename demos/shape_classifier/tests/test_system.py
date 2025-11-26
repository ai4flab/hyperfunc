"""Tests for CLIP classifier system."""

import pytest
import torch

from clip_classifier import (
    CLASS_NAMES,
    CLIPClassifierSystem,
    ProjectionWeights,
    ClassifierWeights,
    accuracy_metric,
    classify,
    generate_synthetic_dataset,
    generate_synthetic_image,
    project_features,
    reset_clip_encoder,
)
from hyperfunc import ESHybridSystemOptimizer


@pytest.fixture(autouse=True)
def reset_encoder():
    """Reset CLIP encoder singleton between tests."""
    reset_clip_encoder()
    yield
    reset_clip_encoder()


def test_generate_synthetic_image():
    """Test synthetic image generation."""
    img = generate_synthetic_image("red_circle")
    assert img.size == (224, 224)
    assert img.mode == "RGB"

    img = generate_synthetic_image("blue_square", size=128)
    assert img.size == (128, 128)


def test_generate_synthetic_dataset():
    """Test synthetic dataset generation."""
    data = generate_synthetic_dataset(n_per_class=2)
    assert len(data) == 20  # 10 classes * 2 per class

    # Check structure
    ex = data[0]
    assert "image" in ex.inputs
    assert "class_idx" in ex.expected
    assert "class_name" in ex.expected


def test_lora_weight_types():
    """Test LoRA weight type configuration."""
    assert ProjectionWeights.shape() == (128, 512)
    assert ProjectionWeights.noise_rank() == 8

    assert ClassifierWeights.shape() == (10, 128)
    assert ClassifierWeights.noise_rank() == 4


def test_project_features_shape():
    """Test projection hyperfunction output shape."""
    # Create a mock input
    features = torch.randn(512)

    # Create system just to register the hyperfunction
    system = CLIPClassifierSystem(class_names=CLASS_NAMES)
    system.register_hyperfunction(
        project_features,
        hp_init=torch.randn(128, 512) * 0.1,
    )

    # Run projection
    projected = project_features(features)
    assert projected.shape == (128,)
    assert (projected >= 0).all()  # ReLU output


def test_classify_shape():
    """Test classification hyperfunction output shape."""
    features = torch.randn(128)

    system = CLIPClassifierSystem(class_names=CLASS_NAMES)
    system.register_hyperfunction(
        classify,
        hp_init=torch.randn(10, 128) * 0.1,
    )

    probs = classify(features)
    assert probs.shape == (10,)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_accuracy_metric():
    """Test accuracy metric calculation."""
    preds = [
        {"class_idx": 0},
        {"class_idx": 1},
        {"class_idx": 2},
    ]
    expected = [
        {"class_idx": 0},
        {"class_idx": 1},
        {"class_idx": 0},  # Wrong
    ]
    acc = accuracy_metric(preds, expected)
    assert acc == pytest.approx(2 / 3)


@pytest.mark.slow
def test_system_end_to_end():
    """Test full system pipeline (requires CLIP download)."""
    # Small dataset
    data = generate_synthetic_dataset(n_per_class=2)

    system = CLIPClassifierSystem(
        class_names=CLASS_NAMES,
        device="cpu",
        system_optimizer=ESHybridSystemOptimizer(
            steps=2,  # Minimal for testing
            pop_size=4,
            sigma=0.1,
            lr=0.1,
        ),
    )

    torch.manual_seed(42)
    system.register_hyperfunction(
        project_features,
        hp_init=torch.randn(128, 512) * 0.1,
    )
    system.register_hyperfunction(
        classify,
        hp_init=torch.randn(10, 128) * 0.1,
    )

    # Should be able to evaluate
    acc = system.evaluate(data, accuracy_metric)
    assert 0 <= acc <= 1

    # Should be able to optimize (even if just 2 steps)
    system.optimize(data, accuracy_metric)

    # Should still work after optimization
    acc_after = system.evaluate(data, accuracy_metric)
    assert 0 <= acc_after <= 1
