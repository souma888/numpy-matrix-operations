"""A small NumPy-backed matrix class with a predictable API."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


class Matrix:
    """Represent a 2-D numeric matrix and expose common linear algebra operations.

    Parameters
    ----------
    values:
        A rectangular, non-empty 2-D sequence or NumPy array.

    Notes
    -----
    The class is a learning project built on NumPy, not a replacement for NumPy.
    """

    def __init__(self, values: Sequence[Sequence[float]] | np.ndarray):
        array = np.asarray(values, dtype=float)

        if array.ndim != 2:
            raise ValueError("A matrix must be two-dimensional.")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("A matrix cannot be empty.")
        if not np.isfinite(array).all():
            raise ValueError("Matrix values must be finite numbers.")

        # Copy the input so changes to the caller's array do not mutate this Matrix.
        self._data = array.copy()

    @property
    def data(self) -> np.ndarray:
        """Return a copy of the underlying NumPy array."""
        return self._data.copy()

    @property
    def shape(self) -> tuple[int, int]:
        """Return (rows, columns)."""
        return self._data.shape

    @property
    def rows(self) -> int:
        return self._data.shape[0]

    @property
    def cols(self) -> int:
        return self._data.shape[1]

    def _coerce_matrix(self, other: Matrix | Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        if isinstance(other, Matrix):
            return other._data
        array = np.asarray(other, dtype=float)
        if array.ndim != 2:
            raise ValueError("The other operand must be a two-dimensional matrix.")
        if not np.isfinite(array).all():
            raise ValueError("Matrix values must be finite numbers.")
        return array

    def __add__(self, other: Matrix | Sequence[Sequence[float]] | np.ndarray) -> Matrix:
        rhs = self._coerce_matrix(other)
        if self._data.shape != rhs.shape:
            raise ValueError(
                f"Addition requires equal shapes; got {self._data.shape} and {rhs.shape}."
            )
        return Matrix(self._data + rhs)

    def __sub__(self, other: Matrix | Sequence[Sequence[float]] | np.ndarray) -> Matrix:
        rhs = self._coerce_matrix(other)
        if self._data.shape != rhs.shape:
            raise ValueError(
                f"Subtraction requires equal shapes; got {self._data.shape} and {rhs.shape}."
            )
        return Matrix(self._data - rhs)

    def __mul__(self, other: float | int | Matrix | Sequence[Sequence[float]] | np.ndarray) -> Matrix:
        """Use scalar multiplication for scalars and elementwise multiplication for matrices."""
        if np.isscalar(other):
            return Matrix(self._data * float(other))
        rhs = self._coerce_matrix(other)
        if self._data.shape != rhs.shape:
            raise ValueError(
                f"Elementwise multiplication requires equal shapes; got {self._data.shape} and {rhs.shape}."
            )
        return Matrix(self._data * rhs)

    def __rmul__(self, other: float | int) -> Matrix:
        if not np.isscalar(other):
            return NotImplemented
        return Matrix(float(other) * self._data)

    def __truediv__(self, other: float | int) -> Matrix:
        if not np.isscalar(other):
            raise TypeError("Matrix division is supported only by a non-zero scalar.")
        scalar = float(other)
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide a matrix by zero.")
        return Matrix(self._data / scalar)

    def __matmul__(self, other: Matrix | Sequence[Sequence[float]] | np.ndarray) -> Matrix:
        rhs = self._coerce_matrix(other)
        if self.cols != rhs.shape[0]:
            raise ValueError(
                f"Matrix multiplication requires A columns == B rows; got {self.shape} and {rhs.shape}."
            )
        return Matrix(self._data @ rhs)

    def transpose(self) -> Matrix:
        return Matrix(self._data.T)

    @property
    def T(self) -> Matrix:
        """Convenient transpose property: matrix.T."""
        return self.transpose()

    def determinant(self) -> float:
        self._require_square("determinant")
        return float(np.linalg.det(self._data))

    def inverse(self) -> Matrix:
        self._require_square("inverse")
        try:
            return Matrix(np.linalg.inv(self._data))
        except np.linalg.LinAlgError as exc:
            raise ValueError("A singular matrix does not have an inverse.") from exc

    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self._data))

    def trace(self) -> float:
        self._require_square("trace")
        return float(np.trace(self._data))

    def norm(self, ord: int | float | str | None = None) -> float:
        return float(np.linalg.norm(self._data, ord=ord))

    def eigenvalues(self) -> np.ndarray:
        self._require_square("eigenvalues")
        values = np.linalg.eigvals(self._data)
        # Return a copy, as with the data property.
        return values.copy()

    def solve(self, b: Sequence[float] | np.ndarray) -> np.ndarray:
        """Solve Ax = b for square A and return x."""
        self._require_square("solve")
        vector = np.asarray(b, dtype=float)
        if vector.ndim not in (1, 2):
            raise ValueError("b must be a vector or a two-dimensional column/matrix.")
        if vector.shape[0] != self.rows:
            raise ValueError(
                f"b must have {self.rows} rows to solve Ax = b; got {vector.shape[0]}."
            )
        if not np.isfinite(vector).all():
            raise ValueError("b must contain only finite numbers.")
        try:
            return np.linalg.solve(self._data, vector)
        except np.linalg.LinAlgError as exc:
            raise ValueError("The system cannot be solved uniquely (the matrix may be singular).") from exc

    def _require_square(self, operation: str) -> None:
        if self.rows != self.cols:
            raise ValueError(f"{operation.capitalize()} requires a square matrix; got {self.shape}.")

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        array = np.asarray(self._data, dtype=dtype)
        if copy is True:
            return array.copy()
        return array

    def __repr__(self) -> str:
        return f"Matrix(shape={self.shape}, data={self._data.tolist()!r})"

    def __str__(self) -> str:
        return np.array2string(self._data, precision=4, suppress_small=True)
