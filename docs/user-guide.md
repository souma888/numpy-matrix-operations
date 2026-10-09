# User Guide

## Installation

Python 3.10+ is required. From the repository root, create a virtual environment and install the package:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

For the optional API and notebooks:

```bash
python -m pip install -e ".[dev,web,notebooks]"
```

## Construct matrices

```python
from matrixlib import Matrix

A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])
```

Inputs must be rectangular, two-dimensional, non-empty, and finite. Values are copied into float64 storage. The `.data` property returns a copy, so mutating it will not change the Matrix object.

## Arithmetic

```python
A + B       # addition; shapes must match
A - B       # subtraction; shapes must match
A * B       # element-wise multiplication; shapes must match
A * 3       # scalar multiplication
3 * A       # scalar multiplication
A / 2       # scalar division
A @ B       # matrix multiplication; A.cols must equal B.rows
A.T         # transpose
```

Do not confuse `*` and `@`: the first is element-wise for two matrices, the second is the row-by-column matrix product.

## Linear algebra

```python
A.determinant()
A.inverse()
A.pseudoinverse()
A.rank()
A.trace()
A.norm()                 # Frobenius norm by default
A.norm(ord=1)
A.condition_number()
A.eigenvalues()
values, vectors = A.eigenvectors()
x = A.solve([5, 11])     # solve A x = b
```

Determinant, inverse, trace, eigenvalue methods, and solve require square matrices. Pseudoinverse supports rectangular matrices. The solver requires a unique solution and raises ValueError for singular systems.

## Understand a solution

```python
import numpy as np
A = Matrix([[3, 1], [1, 2]])
b = np.array([9.0, 8.0])
x = A.solve(b)

print("x:", x)
print("A @ x:", A.data @ x)
print("residual:", A.data @ x - b)
print("residual norm:", np.linalg.norm(A.data @ x - b))
print("condition number:", A.condition_number())
```

A small residual indicates that the computed solution satisfies the equations closely. It does not, by itself, guarantee that the solution is insensitive to perturbations; inspect the condition number as well.

## Optional API

Install the web dependencies and run:

```bash
python -m pip install -e ".[web]"
python -m uvicorn matrixlib.api:app --reload
```

Open http://127.0.0.1:8000/docs for interactive documentation. See the main README for request formats, supported operations, CORS configuration, and size limits.

## Troubleshooting

- **Module not found:** activate the virtual environment and run `python -m pip install -e .` from the repository root.
- **Shape mismatch:** print `.shape` for each matrix. For A @ B, A's number of columns must equal B's number of rows.
- **Singular matrix:** an inverse or unique solution does not exist. Consider the pseudoinverse or least-squares methods.
- **Unexpected complex eigenvalues:** real matrices can have complex eigenvalues.
- **Tiny numerical differences:** use tolerance-based comparisons such as `np.testing.assert_allclose`.
