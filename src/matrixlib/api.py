"""FastAPI adapter for the NumPy Matrix Operations Library.

Run locally with:
    python -m uvicorn matrixlib.api:app --reload
"""
import os
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from matrixlib import Matrix

Operation = Literal[
    "add", "subtract", "multiply", "transpose", "determinant", "inverse",
    "rank", "trace", "norm", "pseudoinverse", "condition_number", "solve",
    "eigenvalues",
]
MAX_DIMENSION = 20

app = FastAPI(
    title="NumPy Matrix Operations API",
    description=(
        "A small educational HTTP API backed by the Matrix class. "
        "All input matrices are limited to 20 rows and 20 columns."
    ),
    version="1.1.0",
)

origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:3000,http://localhost:3000,"
        "http://127.0.0.1:5500,http://localhost:5500",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class OperationRequest(BaseModel):
    """JSON request schema for one matrix operation."""

    operation: Operation
    a: list[list[float]] = Field(min_length=1)
    b: list[list[float]] | None = None


def make_matrix(values: list[list[float]], name: str) -> Matrix:
    """Validate dimensions, rectangular shape and finite values."""
    if not values or not values[0]:
        raise ValueError(f"Matrix {name} cannot be empty.")
    if len(values) > MAX_DIMENSION or len(values[0]) > MAX_DIMENSION:
        raise ValueError(f"Matrices are limited to {MAX_DIMENSION} rows and columns.")
    if any(len(row) != len(values[0]) for row in values):
        raise ValueError(f"Every row in matrix {name} must have equal length.")
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"Matrix {name} must contain finite numbers.")
    return Matrix(array)


def clean_number(value: float) -> float:
    """Convert a scalar to a JSON-safe finite Python float."""
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("The operation produced a non-finite result.")
    return 0.0 if abs(result) < 1e-12 else result


def json_eigenvalues(values: np.ndarray) -> list[float | dict[str, float]]:
    """Encode real eigenvalues as numbers and complex ones as real/imag pairs."""
    encoded: list[float | dict[str, float]] = []
    for value in values:
        if abs(float(np.imag(value))) < 1e-12:
            encoded.append(clean_number(float(np.real(value))))
        else:
            encoded.append({
                "real": clean_number(float(np.real(value))),
                "imag": clean_number(float(np.imag(value))),
            })
    return encoded


@app.get("/")
def home() -> dict[str, str]:
    """Basic service information."""
    return {"name": app.title, "status": "running", "documentation": "/docs"}


@app.get("/health")
def health() -> dict[str, str]:
    """Health-check endpoint."""
    return {"status": "healthy"}


@app.post("/api/operate")
def operate(request: OperationRequest) -> dict[str, object]:
    """Validate and execute a requested operation."""
    try:
        a = make_matrix(request.a, "A")
        operation = request.operation
        result_type: str

        if operation == "transpose":
            result, result_type = a.T, "matrix"
        elif operation == "determinant":
            result, result_type = clean_number(a.determinant()), "scalar"
        elif operation == "inverse":
            result, result_type = a.inverse(), "matrix"
        elif operation == "pseudoinverse":
            result, result_type = a.pseudoinverse(), "matrix"
        elif operation == "rank":
            result, result_type = a.rank(), "scalar"
        elif operation == "trace":
            result, result_type = clean_number(a.trace()), "scalar"
        elif operation == "norm":
            result, result_type = clean_number(a.norm()), "scalar"
        elif operation == "condition_number":
            condition = a.condition_number()
            result = condition if np.isfinite(condition) else "infinity"
            result_type = "scalar"
        elif operation == "eigenvalues":
            result, result_type = json_eigenvalues(a.eigenvalues()), "eigenvalues"
        elif operation == "solve":
            if request.b is None:
                raise ValueError("Operation 'solve' requires right-hand side matrix B.")
            b = make_matrix(request.b, "B")
            result, result_type = Matrix(a.solve(b.data)), "matrix"
        else:
            if request.b is None:
                raise ValueError(f"Operation '{operation}' requires matrix B.")
            b = make_matrix(request.b, "B")
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a @ b
            else:
                raise ValueError(f"Unsupported operation: {operation}.")
            result_type = "matrix"

        result_data = result.data.tolist() if isinstance(result, Matrix) else result
        return {
            "operation": operation,
            "result_type": result_type,
            "result": result_data,
            "message": f"{operation.replace('_', ' ').capitalize()} calculated by Python.",
        }
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except np.linalg.LinAlgError as exc:
        raise HTTPException(
            status_code=422,
            detail="This operation cannot be completed for these matrix values.",
        ) from exc
