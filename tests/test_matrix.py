"""Unit tests for the public Matrix API."""
import numpy as np
import pytest

from matrixlib import Matrix


def test_addition_and_subtraction():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    np.testing.assert_allclose((a + b).data, [[6, 8], [10, 12]])
    np.testing.assert_allclose((b - a).data, [[4, 4], [4, 4]])


def test_scalar_and_elementwise_multiplication_are_distinct():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[2, 3], [4, 5]])
    np.testing.assert_allclose((a * b).data, [[2, 6], [12, 20]])
    np.testing.assert_allclose((a * 2).data, [[2, 4], [6, 8]])
    np.testing.assert_allclose((2 * a).data, [[2, 4], [6, 8]])
    np.testing.assert_allclose((a @ b).data, [[10, 13], [22, 29]])


def test_transpose_rectangular_matrix():
    np.testing.assert_allclose(
        Matrix([[1, 2, 3], [4, 5, 6]]).T.data,
        [[1, 4], [2, 5], [3, 6]],
    )


def test_determinant_inverse_and_identity():
    a = Matrix([[4, 7], [2, 6]])
    assert a.determinant() == pytest.approx(10)
    np.testing.assert_allclose(a.inverse().data, [[0.6, -0.7], [-0.2, 0.4]])
    np.testing.assert_allclose((a @ a.inverse()).data, np.eye(2), atol=1e-12)


def test_singular_inverse_raises():
    with pytest.raises(ValueError, match="singular"):
        Matrix([[1, 2], [2, 4]]).inverse()


def test_rank_trace_norm_and_condition_number():
    a = Matrix([[1, 2], [3, 4]])
    assert a.rank() == 2
    assert a.trace() == pytest.approx(5)
    assert a.norm() == pytest.approx(np.sqrt(30))
    assert a.condition_number() == pytest.approx(np.linalg.cond(a.data))


def test_pseudoinverse_supports_rectangular_matrices():
    a = Matrix([[1, 2, 3], [4, 5, 6]])
    pinv = a.pseudoinverse()
    assert pinv.shape == (3, 2)
    np.testing.assert_allclose((a @ pinv @ a).data, a.data, atol=1e-10)


def test_eigenvalues_and_eigenvectors():
    a = Matrix([[2, 0], [0, 3]])
    values, vectors = a.eigenvectors()
    assert sorted(values.tolist()) == [2, 3]
    np.testing.assert_allclose(a.eigenvalues(), values)
    for index, value in enumerate(values):
        np.testing.assert_allclose(a.data @ vectors[:, index], value * vectors[:, index])


def test_solve_vector_and_multiple_right_hand_sides():
    a = Matrix([[3, 1], [1, 2]])
    np.testing.assert_allclose(a.solve([9, 8]), [2, 3])
    np.testing.assert_allclose(a.solve([[9, 1], [8, 0]]), [[2, 0.4], [3, -0.2]])


def test_shape_and_data_are_defensive():
    source = np.array([[1.0, 2.0], [3.0, 4.0]])
    matrix = Matrix(source)
    source[0, 0] = 999
    assert matrix.data[0, 0] == 1
    exported = matrix.data
    exported[0, 0] = 999
    assert matrix.data[0, 0] == 1
    assert matrix.shape == (2, 2)
    assert matrix.rows == 2
    assert matrix.cols == 2
    assert matrix.is_square
    assert matrix.copy().tolist() == matrix.tolist()


@pytest.mark.parametrize("values", [[], [[]], [[], []], [1, 2], [[1, 2], [3]]])
def test_rejects_empty_non_2d_or_ragged_input(values):
    with pytest.raises(ValueError):
        Matrix(values)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_non_finite_input(value):
    with pytest.raises(ValueError, match="finite"):
        Matrix([[1, value]])


def test_shape_mismatch_errors_are_actionable():
    with pytest.raises(ValueError, match="equal shapes"):
        Matrix([[1, 2]]) + Matrix([[1], [2]])
    with pytest.raises(ValueError, match="columns"):
        Matrix([[1, 2]]) @ Matrix([[1, 2]])


def test_square_only_operations_reject_rectangular_matrices():
    a = Matrix([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="square"):
        a.determinant()
    with pytest.raises(ValueError, match="square"):
        a.inverse()
    with pytest.raises(ValueError, match="square"):
        a.trace()
    with pytest.raises(ValueError, match="square"):
        a.solve([1, 2])


def test_division_validation():
    with pytest.raises(ZeroDivisionError):
        Matrix([[1]]) / 0
    with pytest.raises(ValueError, match="finite"):
        Matrix([[1]]) / float("inf")
    with pytest.raises(TypeError):
        Matrix([[1]]) / Matrix([[2]])


def test_tolerance_arguments_are_validated():
    with pytest.raises(ValueError, match="tol"):
        Matrix([[1]]).rank(tol=-1)
    with pytest.raises(ValueError, match="atol"):
        Matrix([[1]]).is_symmetric(atol=-1)


def test_symmetric_check():
    assert Matrix([[1, 2], [2, 3]]).is_symmetric()
    assert not Matrix([[1, 2], [3, 4]]).is_symmetric()
    assert not Matrix([[1, 2, 3], [4, 5, 6]]).is_symmetric()
