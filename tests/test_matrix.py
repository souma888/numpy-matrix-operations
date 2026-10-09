import numpy as np
import pytest

from matrixlib import Matrix


def test_addition():
    result = Matrix([[1, 2], [3, 4]]) + Matrix([[5, 6], [7, 8]])
    np.testing.assert_allclose(result.data, [[6, 8], [10, 12]])


def test_subtraction():
    result = Matrix([[5, 6], [7, 8]]) - Matrix([[1, 2], [3, 4]])
    np.testing.assert_allclose(result.data, [[4, 4], [4, 4]])


def test_matrix_multiplication():
    result = Matrix([[1, 2], [3, 4]]) @ Matrix([[5, 6], [7, 8]])
    np.testing.assert_allclose(result.data, [[19, 22], [43, 50]])


def test_elementwise_multiplication_and_scalar_multiplication():
    np.testing.assert_allclose((Matrix([[1, 2]]) * Matrix([[3, 4]])).data, [[3, 8]])
    np.testing.assert_allclose((2 * Matrix([[1, 2]])).data, [[2, 4]])


def test_division_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        Matrix([[1]]) / 0


def test_transpose():
    np.testing.assert_allclose(Matrix([[1, 2, 3], [4, 5, 6]]).T.data, [[1, 4], [2, 5], [3, 6]])


def test_determinant_and_inverse():
    a = Matrix([[4, 7], [2, 6]])
    assert a.determinant() == pytest.approx(10)
    np.testing.assert_allclose(a.inverse().data, [[0.6, -0.7], [-0.2, 0.4]])


def test_singular_inverse_raises():
    with pytest.raises(ValueError, match="singular"):
        Matrix([[1, 2], [2, 4]]).inverse()


def test_rank_trace_norm():
    a = Matrix([[1, 2], [3, 4]])
    assert a.rank() == 2
    assert a.trace() == pytest.approx(5)
    assert a.norm() == pytest.approx(np.sqrt(30))


def test_solve():
    x = Matrix([[3, 1], [1, 2]]).solve([9, 8])
    np.testing.assert_allclose(x, [2, 3])


def test_rejects_ragged_input():
    with pytest.raises(ValueError):
        Matrix([[1, 2], [3]])


def test_rejects_non_finite_input():
    with pytest.raises(ValueError, match="finite"):
        Matrix([[1, float("nan")]])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="equal shapes"):
        Matrix([[1, 2]]) + Matrix([[1], [2]])
