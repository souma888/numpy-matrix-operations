# NumPy Matrix Operations Library

A small learning project that wraps common matrix operations in a reusable Python `Matrix` class, using NumPy for numerical computation.

## Features

- Matrix addition and subtraction
- Scalar and elementwise multiplication
- Matrix multiplication using `@`
- Scalar division
- Transpose
- Determinant and inverse
- Rank, trace, norm and eigenvalues
- Solve a square linear system `Ax = b`
- Shape and finite-number validation
- Automated tests with pytest
- Optional FastAPI web API dependencies

## Install for development

Windows CMD, from the project root:

```cmd
py -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,web]"
```

If `.venv` already exists, reuse it. Do not commit `.venv` to Git.

## Run the example

```cmd
.venv\Scripts\python.exe examples\basic_usage.py
```

## Run tests

```cmd
.venv\Scripts\python.exe -m pytest
```

## Quick example

```python
from matrixlib import Matrix

a = Matrix([[1, 2], [3, 4]])
b = Matrix([[5, 6], [7, 8]])

print(a + b)
print(a @ b)
print(a.determinant())
print(a.inverse())
```

This package is an educational wrapper around NumPy, not a replacement for NumPy itself.
