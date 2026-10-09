# HTTP API Reference

The API is an optional FastAPI adapter around the Python Matrix class. Install it with `python -m pip install -e ".[web]"` and start it from the repository root:

```bash
python -m uvicorn matrixlib.api:app --reload
```

Interactive OpenAPI docs are available at `http://127.0.0.1:8000/docs`.

## Endpoints

### `GET /`
Returns service name, status, and the API documentation path.

### `GET /health`
Returns `{"status": "healthy"}`.

### `POST /api/operate`
Request fields:
- `operation`: operation name listed below.
- `a`: required matrix A as a list of rows.
- `b`: optional/required matrix B depending on operation. For `solve`, B is a matrix of right-hand-side columns.

| Operation | Needs B? | Result |
|---|---:|---|
| `add` | Yes | Matrix |
| `subtract` | Yes | Matrix |
| `multiply` | Yes | Matrix product A @ B |
| `transpose` | No | Matrix |
| `determinant` | No | Scalar |
| `inverse` | No | Matrix |
| `pseudoinverse` | No | Matrix |
| `rank` | No | Integer scalar |
| `trace` | No | Scalar |
| `norm` | No | Frobenius norm |
| `condition_number` | No | Scalar or the string `"infinity"` |
| `eigenvalues` | No | Real numbers or complex-value objects |
| `solve` | Yes | Matrix of solution columns |

Example request:

```json
{
  "operation": "solve",
  "a": [[3, 1], [1, 2]],
  "b": [[9], [8]]
}
```

Example response shape:

```json
{
  "operation": "solve",
  "result_type": "matrix",
  "result": [[2.0], [3.0]],
  "message": "Solve calculated by Python."
}
```

For complex eigenvalues, each non-real value is represented as `{"real": ..., "imag": ...}`. Real eigenvalues are represented as JSON numbers.

## Validation and errors

- Matrices must be rectangular, non-empty, finite, and at most 20 × 20.
- Addition and subtraction require equal shapes.
- Matrix multiplication requires A's column count to equal B's row count.
- Square-only operations reject rectangular A.
- Operations requiring B reject requests that omit it.
- Invalid operations and incompatible inputs return HTTP 422 with a human-readable `detail`.

## CORS

The `CORS_ORIGINS` environment variable is a comma-separated list of allowed origins. The default permits common local development origins. Set the exact deployed frontend origin when deploying; avoid permissive wildcard origins.
