"""Deterministic synthetic benchmark for a public chess-vision proof point."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ANCHOR_SQUARES = ("a1", "h1", "a8", "h8", "e1", "e8")
LABELS = ("standard", "rotated", "occluded")


@dataclass(frozen=True)
class BoardSample:
    sample_id: str
    label: str
    rotation_degrees: int
    visible_anchors: tuple[str, ...]

    def feature_row(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "label": self.label,
            "rotation_degrees": str(self.rotation_degrees),
            "visible_anchor_count": str(len(self.visible_anchors)),
            "visible_anchors": " ".join(self.visible_anchors),
        }


def generate_samples(seed: int, samples: int) -> list[BoardSample]:
    rng = random.Random(seed)
    generated: list[BoardSample] = []

    for index in range(samples):
        label = LABELS[index % len(LABELS)]
        if label == "standard":
            rotation = 0
            anchors = ANCHOR_SQUARES
        elif label == "rotated":
            rotation = rng.choice((90, 180, 270))
            anchors = ANCHOR_SQUARES
        else:
            rotation = 0
            hidden_count = rng.choice((1, 2, 3))
            hidden = set(rng.sample(ANCHOR_SQUARES, hidden_count))
            anchors = tuple(square for square in ANCHOR_SQUARES if square not in hidden)

        generated.append(
            BoardSample(
                sample_id=f"synthetic-{index:04d}",
                label=label,
                rotation_degrees=rotation,
                visible_anchors=tuple(anchors),
            )
        )

    return generated


def predict_baseline(sample: BoardSample) -> str:
    if sample.rotation_degrees != 0:
        return "rotated"
    if len(sample.visible_anchors) < len(ANCHOR_SQUARES):
        return "occluded"
    return "standard"


def evaluate(samples: list[BoardSample]) -> dict[str, float | int]:
    correct = sum(1 for sample in samples if predict_baseline(sample) == sample.label)
    return {
        "samples": len(samples),
        "correct": correct,
        "accuracy": correct / len(samples) if samples else 0.0,
    }


def write_csv(samples: list[BoardSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0].feature_row().keys()))
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.feature_row())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic chess-vision benchmark.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=18)
    parser.add_argument("--write-csv", type=Path)
    args = parser.parse_args()

    samples = generate_samples(seed=args.seed, samples=args.samples)
    metrics = evaluate(samples)
    counts = Counter(sample.label for sample in samples)

    if args.write_csv:
        write_csv(samples, args.write_csv)

    print(f"samples={metrics['samples']}")
    print(f"accuracy={metrics['accuracy']:.3f}")
    print("labels=" + ",".join(f"{label}:{counts[label]}" for label in LABELS))


if __name__ == "__main__":
    main()

