#!/usr/bin/env python3
"""Train shape classifier using ES optimization."""

import argparse
import asyncio
import time

from shape_classifier import (
    ShapeClassifierSystem,
    CLASS_NAMES,
    generate_synthetic_dataset,
    accuracy_metric,
)
from hyperfunc import ESHybridSystemOptimizer


async def main():
    parser = argparse.ArgumentParser(description="Train shape classifier with ES")
    parser.add_argument("--steps", type=int, default=50, help="ES optimization steps")
    parser.add_argument("--pop-size", type=int, default=32, help="ES population size")
    parser.add_argument("--sigma", type=float, default=0.1, help="ES noise scale")
    parser.add_argument("--lr", type=float, default=0.1, help="ES learning rate")
    parser.add_argument("--n-per-class", type=int, default=20, help="Examples per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Generate training data (features are pre-extracted for fast training)
    print("Generating synthetic dataset...")
    train_data = generate_synthetic_dataset(
        n_per_class=args.n_per_class,
        seed=args.seed,
    )
    print(f"Generated {len(train_data)} examples across {len(CLASS_NAMES)} classes")
    print(f"Classes: {CLASS_NAMES}")

    # Create system with ES optimizer
    # Hyperfunctions auto-register with Xavier-like initialization when called
    print(f"\nCreating ShapeClassifierSystem...")
    system = ShapeClassifierSystem(
        class_names=CLASS_NAMES,
        system_optimizer=ESHybridSystemOptimizer(
            steps=args.steps,
            pop_size=args.pop_size,
            sigma=args.sigma,
            lr=args.lr,
            antithetic=True,
        ),
    )

    # Evaluate before optimization
    print("\nEvaluating initial model...")
    initial_acc = await system.evaluate(train_data, accuracy_metric)
    print(f"Initial accuracy: {initial_acc:.2%}")

    # Optimize
    print(f"\nOptimizing ({args.steps} steps, pop_size={args.pop_size})...")
    start_time = time.time()
    await system.optimize(train_data, accuracy_metric)
    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.2f}s ({args.steps/elapsed:.2f} steps/sec)")

    # Evaluate after optimization
    print("\nEvaluating optimized model...")
    final_acc = await system.evaluate(train_data, accuracy_metric)
    print(f"Final accuracy: {final_acc:.2%}")
    print(f"Improvement: {final_acc - initial_acc:+.2%}")

    # Show some predictions
    print("\nSample predictions:")
    for i in range(min(10, len(train_data))):
        ex = train_data[i]
        pred = await system.run(**ex.inputs)
        expected = ex.expected["class_name"]
        status = "✓" if pred["class"] == expected else "✗"
        print(f"  {status} Predicted: {pred['class']:15} ({pred['confidence']:.1%}) | Expected: {expected}")


if __name__ == "__main__":
    asyncio.run(main())
