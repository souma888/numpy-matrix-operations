"""Integration tests for the optional FastAPI adapter."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from matrixlib.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_matrix_multiplication_endpoint():
    response = client.post("/api/operate", json={
        "operation": "multiply",
        "a": [[1, 2], [3, 4]],
        "b": [[5, 6], [7, 8]],
    })
    assert response.status_code == 200
    assert response.json()["result"] == [[19.0, 22.0], [43.0, 50.0]]


def test_transpose_endpoint():
    response = client.post("/api/operate", json={
        "operation": "transpose",
        "a": [[1, 2, 3], [4, 5, 6]],
    })
    assert response.status_code == 200
    assert response.json()["result"] == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]


def test_shape_mismatch_returns_422():
    response = client.post("/api/operate", json={
        "operation": "add",
        "a": [[1, 2]],
        "b": [[1], [2]],
    })
    assert response.status_code == 422
    assert "equal shapes" in response.json()["detail"]


def test_solve_endpoint_uses_rhs_columns():
    response = client.post("/api/operate", json={
        "operation": "solve",
        "a": [[3, 1], [1, 2]],
        "b": [[9], [8]],
    })
    assert response.status_code == 200
    assert response.json()["result"] == [[2.0], [3.0]]


def test_complex_eigenvalues_are_json_safe():
    response = client.post("/api/operate", json={
        "operation": "eigenvalues",
        "a": [[0, -1], [1, 0]],
    })
    assert response.status_code == 200
    values = response.json()["result"]
    assert len(values) == 2
    assert all(isinstance(value, dict) for value in values)
    assert all(set(value) == {"real", "imag"} for value in values)


def test_singular_condition_number_is_json_safe():
    response = client.post("/api/operate", json={
        "operation": "condition_number",
        "a": [[1, 2], [2, 4]],
    })
    assert response.status_code == 200
    assert response.json()["result"] == "infinity"


def test_dimension_limit_is_enforced():
    matrix = [[float(i == j) for j in range(21)] for i in range(21)]
    response = client.post("/api/operate", json={
        "operation": "transpose",
        "a": matrix,
    })
    assert response.status_code == 422
