"""HTTP API for the NumPy Matrix Operations Library."""

import os
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from matrixlib import Matrix


Operation = Literal[
    "add", "subtract", "multiply", "transpose",
    "determinant", "inverse", "rank", "trace", "norm"
]

MAX_DIMENSION = 20

app = FastAPI(
    title="NumPy Matrix Operations API",
    description="Runs operations using the Python Matrix library.",
    version="1.0.0",
)


# Only allow browser requests from configured website origins.
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
    operation: Operation
    a: list[list[float]] = Field(min_length=1)
    b: list[list[float]] | None = None


def make_matrix(values: list[list[float]], name: str) -> Matrix:
    """Validate the dimensions and values, then create a Matrix object."""
    if not values or not values[0]:
        raise ValueError(f"Matrix {name} cannot be empty.")

    if len(values) > MAX_DIMENSION or len(values[0]) > MAX_DIMENSION:
        raise ValueError("Matrices are limited to 20 rows and 20 columns.")

    if any(len(row) != len(values[0]) for row in values):
        raise ValueError(f"Every row in matrix {name} must have equal length.")

    array = np.asarray(values, dtype=float)

    if not np.isfinite(array).all():
        raise ValueError(f"Matrix {name} must contain finite numbers.")

    return Matrix(array)


def clean_number(value: float) -> float:
    """Convert a NumPy/Python scalar into a JSON-safe finite float."""
    result = float(value)

    if not np.isfinite(result):
        raise ValueError("The operation produced a non-finite result.")

    return 0.0 if abs(result) < 1e-12 else result


@app.get("/")
def home():
    return {
        "name": "NumPy Matrix Operations API",
        "status": "running",
        "documentation": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/api/operate")
def operate(request: OperationRequest):
    """Execute an operation through the actual Matrix class."""
    try:
        a = make_matrix(request.a, "A")
        operation = request.operation

        if operation == "transpose":
            result = a.transpose()
            result_type = "matrix"

        elif operation == "determinant":
            result = clean_number(a.determinant())
            result_type = "scalar"

        elif operation == "inverse":
            result = a.inverse()
            result_type = "matrix"

        elif operation == "rank":
            result = a.rank()
            result_type = "scalar"

        elif operation == "trace":
            result = clean_number(a.trace())
            result_type = "scalar"

        elif operation == "norm":
            result = clean_number(a.norm())
            result_type = "scalar"

        else:
            if request.b is None:
                raise ValueError(
                    f"Operation '{operation}' requires matrix B."
                )

            b = make_matrix(request.b, "B")

            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a @ b
            else:
                raise ValueError("Unsupported operation.")

            result_type = "matrix"

        # Convert the result into standard Python values for JSON.
        if result_type == "matrix":
            result_data = result.data.tolist()
        else:
            result_data = result

        return {
            "operation": operation,
            "result_type": result_type,
            "result": result_data,
            "message": f"{operation.capitalize()} calculated by Python.",
        }

    except (ValueError, TypeError, ZeroDivisionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except np.linalg.LinAlgError as exc:
        raise HTTPException(
            status_code=422,
            detail="This operation cannot be completed for these matrix values.",
        ) from exc