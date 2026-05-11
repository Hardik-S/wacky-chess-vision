# Wacky Chess Vision

This public repo packages a lightweight, reproducible benchmark for the resume claim that a chess-vision model can classify unusual board states. The first proof point is intentionally small: deterministic synthetic data generation, a baseline classifier, and tests that make the benchmark contract reviewable without GPU access.

## What This Proves

- ML experiment hygiene: dataset generation, labels, metrics, and test expectations are separated.
- Reproducibility: the smoke benchmark is deterministic from a documented seed.
- Recruiter readability: the repo explains what is proven now and what remains future work.
- Public safety: no scraped tournament images, private datasets, or third-party assets are included.

## File Structure

- `src/wacky_chess_vision.py` contains the synthetic board generator and baseline evaluator.
- `data/sample_labels.csv` is a tiny public-safe fixture for inspection.
- `tests/test_wacky_chess_vision.py` verifies generation and baseline behavior.
- `docs/BENCHMARK_PROTOCOL.md` documents decisions, rejected shortcuts, and next experiments.

## Current Benchmark

The synthetic task classifies whether a board position is `standard`, `rotated`, or `occluded`. The baseline does not claim deep-learning performance. It exists so future CNN or domain-adaptation work has a stable, testable entry point.

The CLI can now export a deterministic text board snapshot. This is intentionally lighter than PNG generation, but it makes the synthetic vision contract visible before adding image dependencies.

Run:

```powershell
python -m unittest discover -s tests
python src\wacky_chess_vision.py --seed 42 --samples 18
python src\wacky_chess_vision.py --seed 42 --samples 18 --write-snapshot artifacts\board_snapshot.txt
```

## Next Work

The next meaningful improvement is a tiny image renderer that exports generated boards as PNG fixtures. The text snapshot added here is the prerequisite contract: it defines which anchors are visible, hidden, and reviewable before image assets exist.

