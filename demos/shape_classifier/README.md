# CLIP Image Classifier Demo

This demo shows how to use **hyperfunc** to train a CLIP-based image classifier using Evolution Strategies (ES) with low-rank noise.

## Architecture

```
Image → CLIP ViT-B/32 → LoRA Projection → Classification Head → Prediction
         (frozen)        (512→128)          (128→10)
         ~150MB          trainable          trainable
```

- **CLIP ViT-B/32**: Pretrained vision encoder (frozen)
- **LoRA Projection**: 512→128 linear layer with ReLU, optimized via ES
- **Classification Head**: 128→10 linear layer with softmax, optimized via ES

Both trainable layers use `LoRAWeight` which enables efficient ES optimization with low-rank noise.

## Installation

```bash
cd demos/clip_classifier
uv venv .venv
source .venv/bin/activate
uv sync
```

This will install hyperfunc (from the parent directory) and all dependencies.

## Usage

### Train on Synthetic Data

The synthetic dataset contains colored shapes (5 colors × 2 shapes = 10 classes):

```bash
python scripts/train.py --steps 50 --pop-size 32
```

Options:
- `--steps`: Number of ES optimization steps (default: 50)
- `--pop-size`: ES population size (default: 32)
- `--sigma`: ES noise scale (default: 0.1)
- `--lr`: ES learning rate (default: 0.1)
- `--n-per-class`: Training examples per class (default: 20)
- `--device`: Device to use (cpu/cuda/mps)

### Evaluate

```bash
python scripts/evaluate.py --dataset synthetic
python scripts/evaluate.py --dataset cifar10
```

### Run Tests

```bash
pytest tests/ -v
```

Note: Tests marked `@pytest.mark.slow` require downloading CLIP (~150MB).

## How It Works

### HyperSystem

The `CLIPClassifierSystem` subclasses `HyperSystem` and implements `run()`:

```python
class CLIPClassifierSystem(HyperSystem):
    def run(self, image) -> dict:
        # 1. Extract CLIP features (frozen)
        features = self.clip.encode_image(image)

        # 2. Project through LoRA layer (trainable)
        projected = project_features(features)

        # 3. Classify (trainable)
        probs = classify(projected)

        return {"class": ..., "confidence": ..., "probs": ...}
```

### LoRA Weights

Both trainable layers use `LoRAWeight.create()`:

```python
ProjectionWeights = LoRAWeight.create(out_dim=128, in_dim=512, noise_rank=8)
ClassifierWeights = LoRAWeight.create(out_dim=10, in_dim=128, noise_rank=4)
```

This creates hp_types that:
- Store 2D weight matrices
- Use low-rank noise during ES optimization (more efficient than full-rank)

### ES Optimization

The `ESHybridSystemOptimizer` uses:
- Antithetic sampling (pairs of +noise/-noise for lower variance)
- Low-rank noise for 2D parameters (Eggroll-style)
- Proper gradient-based ES (fitness-weighted noise average)

## Results

### CIFAR-10 (recommended)

CIFAR-10 works well with CLIP since it contains natural images:
- ~10% accuracy initially (random chance)
- ~50% accuracy after 100 ES steps (with 200 examples)

This is a 5x improvement from random, demonstrating effective ES optimization.

### Synthetic Colored Shapes

The synthetic dataset (colored shapes on white background) has limited accuracy (~60%) because:
- CLIP was trained on natural images, not simple geometric shapes
- CLIP embeddings for different colored shapes are very similar (0.83-0.94 cosine similarity)
- CLIP focuses on semantic content rather than low-level color/shape features

The synthetic dataset is useful for quick testing but doesn't showcase CLIP's strengths.

The improvement demonstrates that ES can effectively optimize the LoRA projection and classification layers without backpropagation.

## Files

```
clip_classifier/
├── __init__.py          # Package exports
├── models.py            # CLIP encoder wrapper
├── data.py              # Synthetic and CIFAR-10 data loaders
└── system.py            # HyperSystem implementation

scripts/
├── train.py             # Training script
└── evaluate.py          # Evaluation script

tests/
└── test_system.py       # Unit tests
```
