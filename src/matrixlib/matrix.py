"""A small, predictable matrix wrapper built on NumPy.

This class is intended for learning and small experiments. NumPy/SciPy remain
the recommended tools for high-performance numerical computing.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Matrix:
    """Represent a non-empty, finite, two-dimensional real matrix.

    Inputs are copied and converted to float64. The public data property also
    returns a copy, so callers cannot accidentally mutate internal state.
    """

    def __init__(self, values: ArrayLike):
        try:
            raw = np.asarray(values)
            if np.iscomplexobj(raw):
                raise ValueError("Complex matrix values are not supported; provide real numbers.")
            array = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and "Complex matrix values" in str(exc):
                raise
            raise ValueError("Matrix values must form a rectangular numeric array.") from exc

        if array.ndim != 2:
            raise ValueError("A matrix must be two-dimensional.")
        if 0 in array.shape:
            raise ValueError("A matrix cannot have zero rows or zero columns.")
        if not np.isfinite(array).all():
            raise ValueError("Matrix values must be finite numbers.")
        self._data: NDArray[np.float64] = array.copy()

    @property
    def data(self) -> NDArray[np.float64]:
        """Return a defensive copy of the matrix values."""
        return self._data.copy()

    @property
    def shape(self) -> tuple[int, int]:
        """Return (rows, columns)."""
        return self._data.shape

    @property
    def rows(self) -> int:
        """Number of rows."""
        return self._data.shape[0]

    @property
    def cols(self) -> int:
        """Number of columns."""
        return self._data.shape[1]

    @property
    def T(self) -> Matrix:
        """Return the transpose."""
        return self.transpose()

    @property
    def is_square(self) -> bool:
        """Whether the matrix has equal numbers of rows and columns."""
        return self.rows == self.cols

    def copy(self) -> Matrix:
        """Return an independent copy of this matrix."""
        return Matrix(self._data)

    def tolist(self) -> list[list[float]]:
        """Return matrix values as nested Python lists."""
        return self._data.tolist()

    def _coerce_matrix(self, other: Matrix | ArrayLike) -> NDArray:
        if isinstance(other, Matrix):
            return other._data
        try:
            raw = np.asarray(other)
            if np.iscomplexobj(raw):
                raise ValueError("Complex matrix values are not supported; provide real numbers.")
            array = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and "Complex matrix values" in str(exc):
                raise
            raise ValueError("The other operand must be a rectangular numeric matrix.") from exc
        if array.ndim != 2:
            raise ValueError("The other operand must be a two-dimensional matrix.")
        if not np.isfinite(array).all():
            raise ValueError("Matrix values must be finite numbers.")
        return array

    def __add__(self, other: Matrix | ArrayLike) -> Matrix:
        rhs = self._coerce_matrix(other)
        if self.shape != rhs.shape:
            raise ValueError(f"Addition requires equal shapes; got {self.shape} and {rhs.shape}.")
        return Matrix(self._data + rhs)

    def __sub__(self, other: Matrix | ArrayLike) -> Matrix:
        rhs = self._coerce_matrix(other)
        if self.shape != rhs.shape:
            raise ValueError(f"Subtraction requires equal shapes; got {self.shape} and {rhs.shape}.")
        return Matrix(self._data - rhs)

    def __mul__(self, other: float | int | Matrix | ArrayLike) -> Matrix:
        """Multiply by a scalar or do element-wise multiplication; use @ for A times B."""
        if np.isscalar(other):
            return Matrix(self._data * float(other))
        rhs = self._coerce_matrix(other)
        if self.shape != rhs.shape:
            raise ValueError(
                "Elementwise multiplication requires equal shapes; "
                f"got {self.shape} and {rhs.shape}."
            )
        return Matrix(self._data * rhs)

    def __rmul__(self, other: float | int) -> Matrix:
        if not np.isscalar(other):
            return NotImplemented
        return Matrix(float(other) * self._data)

    def __truediv__(self, other: float | int) -> Matrix:
        if not np.isscalar(other):
            raise TypeError("Matrix division is supported only by a scalar.")
        scalar = float(other)
        if not np.isfinite(scalar):
            raise ValueError("The divisor must be finite.")
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide a matrix by zero.")
        return Matrix(self._data / scalar)

    def __matmul__(self, other: Matrix | ArrayLike) -> Matrix:
        rhs = self._coerce_matrix(other)
        if self.cols != rhs.shape[0]:
            raise ValueError(
                "Matrix multiplication requires A columns == B rows; "
                f"got {self.shape} and {rhs.shape}."
            )
        return Matrix(self._data @ rhs)

    def transpose(self) -> Matrix:
        """Return the transpose."""
        return Matrix(self._data.T)

    def determinant(self) -> float:
        """Return the determinant of a square matrix."""
        self._require_square("determinant")
        return float(np.linalg.det(self._data))

    def inverse(self) -> Matrix:
        """Return the inverse, raising ValueError when the matrix is singular."""
        self._require_square("inverse")
        try:
            return Matrix(np.linalg.inv(self._data))
        except np.linalg.LinAlgError as exc:
            raise ValueError("A singular matrix does not have an inverse.") from exc

    def pseudoinverse(self, rcond: float | None = None) -> Matrix:
        """Return the Moore-Penrose pseudoinverse, including for rectangular matrices."""
        if rcond is not None and (not np.isfinite(rcond) or rcond < 0):
            raise ValueError("rcond must be a finite, non-negative number or None.")
        if rcond is None:
            result = np.linalg.pinv(self._data)
        else:
            result = np.linalg.pinv(self._data, rcond=rcond)
        return Matrix(result)

    def rank(self, tol: float | None = None) -> int:
        """Return numerical rank, optionally using an absolute tolerance."""
        if tol is not None and (not np.isfinite(tol) or tol < 0):
            raise ValueError("tol must be a finite, non-negative number or None.")
        return int(np.linalg.matrix_rank(self._data, tol=tol))

    def trace(self) -> float:
        """Return the sum of the main diagonal; requires a square matrix."""
        self._require_square("trace")
        return float(np.trace(self._data))

    def norm(self, ord: int | float | str | None = None) -> float:
        """Return a NumPy matrix norm; ord=None gives the Frobenius norm."""
        return float(np.linalg.norm(self._data, ord=ord))

    def condition_number(self, p: int | float | str | None = None) -> float:
        """Return the condition number; singular matrices have infinite condition."""
        return float(np.linalg.cond(self._data, p=p))

    def eigenvalues(self) -> NDArray:
        """Return eigenvalues of a square matrix (possibly complex)."""
        self._require_square("eigenvalues")
        return np.linalg.eigvals(self._data).copy()

    def eigenvectors(self) -> tuple[NDArray, NDArray]:
        """Return eigenvalues and eigenvectors; eigenvectors are columns."""
        self._require_square("eigenvectors")
        values, vectors = np.linalg.eig(self._data)
        return values.copy(), vectors.copy()

    def solve(self, b: ArrayLike) -> NDArray:
        """Solve Ax=b for square A; b may be a vector or multiple RHS columns."""
        self._require_square("solve")
        try:
            raw = np.asarray(b)
            if np.iscomplexobj(raw):
                raise ValueError("b must contain real numeric values.")
            vector = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and "real numeric values" in str(exc):
                raise
            raise ValueError("b must contain numeric values.") from exc
        if vector.ndim not in (1, 2):
            raise ValueError("b must be a vector or a two-dimensional array.")
        if vector.shape[0] != self.rows:
            raise ValueError(f"b must have {self.rows} rows; got {vector.shape[0]}.")
        if not np.isfinite(vector).all():
            raise ValueError("b must contain only finite numbers.")
        try:
            return np.linalg.solve(self._data, vector)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "The system cannot be solved uniquely; the matrix may be singular."
            ) from exc

    def is_symmetric(self, *, atol: float = 1e-10) -> bool:
        """Return whether the matrix is square and symmetric within atol."""
        if not np.isfinite(atol) or atol < 0:
            raise ValueError("atol must be a finite, non-negative number.")
        return self.is_square and bool(np.allclose(self._data, self._data.T, atol=atol, rtol=0))

    def _require_square(self, operation: str) -> None:
        if not self.is_square:
            raise ValueError(f"{operation.capitalize()} requires a square matrix; got {self.shape}.")

    def __array__(self, dtype=None, copy=None) -> NDArray:
        """Convert to NumPy; copy by default, or expose a view if copy=False is explicit."""
        if copy is False:
            if dtype is not None and np.dtype(dtype) != self._data.dtype:
                raise ValueError("copy=False cannot be honored when a dtype conversion is needed.")
            return self._data
        return np.array(self._data, dtype=dtype, copy=True)

    def __repr__(self) -> str:
        return f"Matrix(shape={self.shape}, data={self._data.tolist()!r})"

    def __str__(self) -> str:
        return np.array2string(self._data, precision=4, suppress_small=True)
