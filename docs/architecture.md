# Architecture

- `src/matrixlib/matrix.py`: the `Matrix` wrapper and its public operations.
- `src/matrixlib/__init__.py`: public import surface.
- `tests/`: automated correctness and validation tests.
- `examples/`: small runnable scripts.
- `notebooks/`: optional exploratory explanations and demonstrations.
- `benchmarks/`: reproducible timing experiments.

The implementation delegates core numerical operations to NumPy rather than reimplementing optimized linear algebra algorithms. The wrapper provides a consistent API, shape checks, and error messages.
