# User Guide

## Installation

From the repository root, create a virtual environment and install the package in editable mode:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Create and use matrices

```python
from matrixlib import Matrix

a = Matrix([[1, 2], [3, 4]])
b = Matrix([[5, 6], [7, 8]])

sum_matrix = a + b
difference = a - b
elementwise_product = a * b
matrix_product = a @ b
transpose = a.T
det = a.determinant()
inverse = a.inverse()
```

## Current behavior

- Matrix input must be a non-empty, two-dimensional array of finite numeric values.
- Addition and subtraction require identical shapes.
- Matrix multiplication requires the number of columns in the left matrix to equal the number of rows in the right matrix.
- Determinant, inverse, trace, eigenvalues, and solving a system require a square matrix.
- Singular matrices cannot be inverted or used to obtain a unique solution.
- Internally, inputs are converted to floating-point NumPy arrays.

This API is an initial version and may evolve; check the changelog for changes.
