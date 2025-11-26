#!/usr/bin/env python3
"""Evaluate shape classifier on test data."""

import argparse
import asyncio

from shape_classifier import (
    ShapeClassifierSystem,
    CLASS_NAMES,
    generate_synthetic_dataset,
    accuracy_metric,
)


async def main():
    parser = argparse.ArgumentParser(description="Evaluate shape classifier")
    parser.add_argument("--n-per-class", type=int, default=10, help="Examples per class")
    parser.add_argument("--seed", type=int, default=123, help="Random seed (different from train)")
    args = parser.parse_args()

    # Load test data
    print("Loading synthetic dataset...")
    test_data = generate_synthetic_dataset(
        n_per_class=args.n_per_class,
        seed=args.seed,
    )
    print(f"Loaded {len(test_data)} test examples")
    print(f"Classes: {CLASS_NAMES}")

    # Create system (no optimizer needed for evaluation)
    # Hyperfunctions auto-register with Xavier-like initialization
    print("\nCreating ShapeClassifierSystem...")
    system = ShapeClassifierSystem(class_names=CLASS_NAMES)

    # Evaluate (random init - just for demo; real usage would load trained weights)
    print("\nEvaluating...")
    acc = await system.evaluate(test_data, accuracy_metric)
    print(f"Accuracy: {acc:.2%}")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    class_correct = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}

    for ex in test_data:
        pred = await system.run(**ex.inputs)
        expected_idx = ex.expected["class_idx"]
        class_total[expected_idx] += 1
        if pred["class_idx"] == expected_idx:
            class_correct[expected_idx] += 1

    for i, name in enumerate(CLASS_NAMES):
        total = class_total[i]
        correct = class_correct[i]
        class_acc = correct / total if total > 0 else 0
        print(f"  {name:15}: {class_acc:.1%} ({correct}/{total})")


if __name__ == "__main__":
    asyncio.run(main())
