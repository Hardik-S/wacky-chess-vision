# Benchmark Protocol

## Scope

This benchmark is a first public artifact for an ML side project. It validates data-contract discipline, not model maturity. The repo currently avoids external dependencies so it can run quickly in a fresh Windows checkout.

## Synthetic Labels

- `standard`: normal board orientation with all anchor pieces visible.
- `rotated`: orientation flag is changed to simulate camera or board rotation.
- `occluded`: one or more anchor squares are hidden to simulate partial vision.

## Decisions

The generator uses structured records rather than image files in this first pass. That keeps verification cheap and makes the board-state assumptions inspectable. A future renderer can convert these records into images once the label protocol is stable.

The baseline classifier is rule-based on purpose. It establishes a minimum benchmark and catches accidental label drift before more complex models are added.

## Rejected Approaches

- Downloading public chess photos was rejected because licensing and annotation quality would be unclear.
- Training a CNN immediately was rejected because it would obscure dataset assumptions.
- Publishing private experiment notes was rejected because the public repo should stand alone.

## Verification Contract

Every run should pass:

```powershell
python -m unittest discover -s tests
python src\wacky_chess_vision.py --seed 42 --samples 18
```

The CLI reports label counts and baseline accuracy for the deterministic fixture.

