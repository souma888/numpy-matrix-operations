# Architecture

## Package boundaries

- `src/matrixlib/matrix.py`: the public `Matrix` class, input validation, operators, and linear algebra methods.
- `src/matrixlib/__init__.py`: stable public import (`from matrixlib import Matrix`).
- `src/matrixlib/api.py`: optional FastAPI adapter. It validates JSON input, invokes the Matrix class, and serializes results.
- `tests/`: automated unit tests.
- `examples/`: small runnable scripts.
- `notebooks/`: educational walkthroughs and visual experiments; notebooks are not the canonical implementation.
- `docs/`: user guide, mathematics and architecture.
- `benchmarks/`: reproducible performance experiments.

## Design principles

1. **Small public API:** a Matrix object wraps one 2D real-valued NumPy array.
2. **Validation at boundaries:** constructor, operators and API input validate shape and finite values.
3. **Defensive data access:** constructor copies values and the `.data` property returns a copy.
4. **Explicit multiplication semantics:** `*` means scalar/element-wise multiplication; `@` means matrix multiplication.
5. **Numerical honesty:** rank, condition numbers, eigen-analysis and other floating-point calculations inherit numerical limitations from NumPy.
6. **Separation of concerns:** HTTP and notebook code call the package instead of duplicating the matrix implementation.

## Computation strategy

The project delegates core numerical algorithms to NumPy. It does not claim that its own code implements optimized matrix multiplication, LU, QR, SVD, or eigenvalue algorithms from scratch. Educational algorithm implementations, if added later, should live in separate modules and be tested against reference implementations.

## API boundaries

The optional HTTP API limits each input matrix to at most 20 rows and 20 columns. This is an application-level guardrail for a public demo, not a limitation of the Python Matrix class. The API returns HTTP 422 for invalid operations and documents itself using FastAPI's OpenAPI support.

## Adding a feature

1. Add or update the public method in `matrix.py`.
2. Add unit tests for ordinary inputs, invalid inputs, and numerical edge cases.
3. Update the API only if the operation belongs in the web service.
4. Update the README API table and user guide.
5. Add an explanatory notebook example where it improves understanding.
6. Add a changelog entry when behavior changes.
