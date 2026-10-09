# NumPy Matrix Operations Library

A small educational Python package that wraps common matrix operations in a reusable `Matrix` class, using NumPy for numerical computation. The project is designed to help you understand matrix APIs, numerical validation, testing, and the relationship between linear algebra and Python software engineering.

> **Scope:** this is a learning wrapper around NumPy, not a replacement for NumPy or SciPy. Most arithmetic and linear algebra algorithms are delegated to NumPy. The value of this project is its consistent API, validation, tests, examples, API adapter, and explanations.

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Mathematical notes](#mathematical-notes)
- [Jupyter notebooks](#jupyter-notebooks)
- [Optional HTTP API](#optional-http-api)
- [Testing and code quality](#testing-and-code-quality)
- [Repository layout](#repository-layout)
- [Design choices and limitations](#design-choices-and-limitations)
- [Contributing](#contributing)
- [License](#license)

## Features

- Matrix construction and shape inspection
- Matrix addition and subtraction
- Scalar multiplication and division
- Element-wise multiplication with `*`
- Matrix multiplication with `@`
- Transpose using `.T` or `.transpose()`
- Determinant and inverse for square matrices
- Moore–Penrose pseudoinverse for rectangular or rank-deficient matrices
- Numerical rank, trace, matrix norms, and condition number
- Eigenvalues and eigenvectors
- Solving square systems `Ax = b`, including multiple right-hand sides
- Symmetry check with configurable absolute tolerance
- Defensive copies for construction and `.data`
- Input validation for dimensionality, rectangular shape, finite values, and operation compatibility
- Automated tests, examples, educational notebooks, and optional FastAPI service

## Installation

Python 3.10 or newer is required. Commands below are shown for Windows PowerShell and can be adapted for macOS/Linux.

### 1. Clone the repository

```bash
git clone https://github.com/souma888/numpy-matrix-operations.git
cd numpy-matrix-operations
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the package

Core package only:

```bash
python -m pip install -e .
```

Development and tests:

```bash
python -m pip install -e ".[dev]"
```

All optional dependencies for the API and notebooks:

```bash
python -m pip install -e ".[dev,web,notebooks]"
```

## Quick start

```python
import numpy as np
from matrixlib import Matrix

A = Matrix([[4, 7], [2, 6]])
B = Matrix([[1, 2], [3, 4]])

print("A =", A)
print("shape:", A.shape)          # (2, 2)
print("A + B =", A + B)
print("A - B =", A - B)
print("A * B (element-wise) =", A * B)
print("A @ B (matrix product) =", A @ B)
print("transpose =", A.T)
print("determinant =", A.determinant())
print("inverse =", A.inverse())
print("rank =", A.rank())
print("trace =", A.trace())
print("Frobenius norm =", A.norm())
print("condition number =", A.condition_number())
print("eigenvalues =", A.eigenvalues())
print("pseudoinverse =", A.pseudoinverse())

b = np.array([1.0, 0.0])
x = A.solve(b)
print("solution x =", x)
print("residual =", A.data @ x - b)
```

## API reference

All matrix inputs are real-valued, finite, non-empty, two-dimensional arrays. Values are converted to `float64`.

| API | Meaning | Requirements / notes |
|---|---|---|
| `Matrix(values)` | Construct a matrix | Rectangular, non-empty 2D input |
| `.data` | Return a copy of values | Mutating the returned array does not mutate the object |
| `.shape`, `.rows`, `.cols` | Dimensions | Work for rectangular matrices |
| `.is_square` | Check whether matrix is square | Boolean |
| `.copy()`, `.tolist()` | Copy or export values | Nested Python list from `.tolist()` |
| `A + B`, `A - B` | Addition and subtraction | Same shape required |
| `A * B` | Element-wise multiplication | Same shape required |
| `A * scalar`, `scalar * A` | Scalar multiplication | Scalar must be finite for useful finite output |
| `A / scalar` | Scalar division | Zero divisor raises `ZeroDivisionError` |
| `A @ B` | Matrix multiplication | `A.cols == B.rows` |
| `A.T`, `A.transpose()` | Transpose | Any non-empty matrix |
| `A.determinant()` | Determinant | Square matrix |
| `A.inverse()` | Matrix inverse | Square and nonsingular |
| `A.pseudoinverse(rcond=None)` | Moore–Penrose pseudoinverse | Rectangular and rank-deficient matrices supported |
| `A.rank(tol=None)` | Numerical rank | Optional absolute tolerance |
| `A.trace()` | Main diagonal sum | Square matrix |
| `A.norm(ord=None)` | Matrix norm | Defaults to Frobenius norm; follows NumPy's supported matrix norms |
| `A.condition_number(p=None)` | Condition number | Infinite for singular matrices |
| `A.eigenvalues()` | Eigenvalues | Square matrix; outputs may be complex |
| `A.eigenvectors()` | Tuple of eigenvalues and eigenvector matrix | Eigenvectors are columns; outputs may be complex |
| `A.solve(b)` | Solve `Ax=b` | Square nonsingular A; b has shape (n,) or (n,k) |

### Element-wise multiplication versus matrix multiplication

The symbols deliberately mean different things:

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

A * B   # [[ 5, 12], [21, 32]]: element-by-element
A @ B   # [[19, 22], [43, 50]]: row-by-column matrix product
```

### Error behavior

- Invalid dimensionality, ragged rows, empty matrices, non-finite input, and incompatible shapes raise `ValueError`.
- Division by zero raises `ZeroDivisionError`.
- Dividing a matrix by another matrix raises `TypeError`; matrix division is not defined as an operator here.
- Square-only operations reject rectangular matrices.
- Inverting or uniquely solving a singular matrix raises `ValueError`.
- Floating-point operations may fail for extreme values if the result is not finite; the constructor rejects non-finite matrix results.

## Mathematical notes

- Matrix multiplication is not generally commutative: `A @ B` may differ from `B @ A`.
- The determinant is defined only for square matrices.
- A nonzero determinant characterizes invertibility in exact arithmetic; floating-point determinant values alone are not a reliable singularity test.
- Rank and condition number are numerical quantities affected by floating-point precision and tolerances.
- A small residual does not necessarily imply a small solution error for an ill-conditioned system.
- Eigenvalues and eigenvectors may be complex even when the input matrix is real.
- The pseudoinverse is useful for rectangular or rank-deficient matrices and least-squares problems.

See [Mathematical Foundations](docs/mathematics.md) and [User Guide](docs/user-guide.md) for longer explanations.

## Jupyter notebooks

The `notebooks/` directory contains guided notebooks that use this package and NumPy to explain the operations with executable examples, visualizations, and exercises.

1. **01 — Getting started and matrix arithmetic:** construction, dimensions, arithmetic, shape errors, transpose, determinant, inverse, rank, trace and norms.
2. **02 — Linear systems and numerical diagnostics:** solving `Ax=b`, residuals, condition numbers, singular systems, pseudoinverses, eigenvalues and eigenvectors.
3. **03 — Visualizing matrices and transformations:** heatmaps, 2D vectors, rotations, scaling, shear and determinant/area relationships.
4. **04 — Testing and benchmarking:** numerical tolerances, reference comparisons, reproducible timing methodology, and why benchmarks must report their environment.

Install notebook dependencies with `python -m pip install -e ".[notebooks,dev]"`, then launch `jupyter lab`. In Google Colab, upload a notebook and install the package from the repository, or follow the notebook's setup cell.

## Optional HTTP API

Install the web extra:

```bash
python -m pip install -e ".[web]"
```

Run from the repository root:

```bash
python -m uvicorn matrixlib.api:app --reload
```

- Service root: http://127.0.0.1:8000/
- Health check: http://127.0.0.1:8000/health
- Interactive API docs: http://127.0.0.1:8000/docs

Example request:

```json
{
  "operation": "multiply",
  "a": [[1, 2], [3, 4]],
  "b": [[5, 6], [7, 8]]
}
```

The API accepts matrices up to 20 rows and 20 columns. Operations requiring matrix B must include it. The `solve` endpoint treats B as one or more right-hand-side columns. For `eigenvalues`, real values are returned as numbers and complex values as objects with `real` and `imag` fields. The condition number is returned as the string `"infinity"` for singular matrices so the response remains JSON-safe.

Configure allowed browser origins with the comma-separated environment variable `CORS_ORIGINS`. Do not use a wildcard origin with credentials for a public deployment.

## Testing and code quality

Run the test suite:

```bash
python -m pytest
```

Run with coverage:

```bash
python -m pytest --cov=matrixlib --cov-report=term-missing
```

Check style:

```bash
ruff check .
```

Tests should use tolerance-based comparisons for floating-point outputs. Do not publish test counts, coverage percentages, or benchmark results until they have been measured on the current revision.

## Repository layout

```text
src/matrixlib/       Matrix class and optional FastAPI adapter
tests/               Automated tests
examples/            Runnable Python examples
notebooks/           Educational Jupyter notebooks
docs/                User guide, mathematics, architecture
benchmarks/          Reproducible benchmark experiments
pyproject.toml       Package metadata and optional dependencies
requirements.txt     Minimal NumPy requirement for simple environments
```

## Design choices and limitations

- This is a dense, real-valued matrix wrapper. Complex outputs are supported for eigen-analysis, but constructing a Matrix from complex input is not supported.
- The implementation delegates numerical algorithms to NumPy. It does not claim to implement LU, QR, SVD, or matrix multiplication algorithms from scratch.
- The API is intended for small educational matrices and imposes size limits.
- The package currently focuses on a small set of common operations rather than every operation available in NumPy/SciPy.
- Performance results depend on the machine, Python, NumPy build, BLAS/LAPACK implementation, and matrix size.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Add tests for behavior changes, run the test suite, update the documentation, and include benchmark methodology for performance claims.

## License

See [LICENSE](LICENSE).
