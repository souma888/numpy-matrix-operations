# Educational notebooks

These notebooks are guided lessons, not the canonical implementation. The package code remains in `src/matrixlib/`; notebooks demonstrate its public API and use NumPy/Matplotlib for independent checks and visualizations.

## Notebooks

| Notebook | Topics |
|---|---|
| [01 — Getting started and matrix arithmetic](01_getting_started_and_matrix_arithmetic.ipynb) | Matrix construction, shapes, arithmetic, transpose, determinant, inverse, properties and heatmaps |
| [02 — Linear systems and numerical diagnostics](02_linear_systems_and_numerical_diagnostics.ipynb) | Solving Ax=b, residuals, conditioning, pseudoinverse, multiple right-hand sides and eigenpairs |
| [03 — Matrix visualization and transformations](03_matrix_visualization_and_transformations.ipynb) | Heatmaps, 2D grids, rotation, scaling, reflection, shear, determinant and area |
| [04 — Testing and benchmarking](04_testing_and_benchmarking.ipynb) | Floating-point tolerances, invariants, reproducible timing and careful interpretation |

## Run locally

From the repository root:

```bash
python -m pip install -e ".[dev,notebooks]"
jupyter lab
```

Open a notebook from the Jupyter file browser and run cells in order.

## Run in Google Colab

1. Open the notebook's GitHub page and copy its URL.
2. In Colab, choose **File → Open notebook → GitHub** and paste the URL, or use the GitHub picker.
3. Before the tutorial cells, clone this working branch and install the package:

```python
!git clone -b docs/notebooks-complete-audit https://github.com/souma888/numpy-matrix-operations.git
%pip install -e ./numpy-matrix-operations
%pip install matplotlib
```

After the branch is merged into `main`, replace the branch name with `main`.

## Notebook conventions

- Run cells from top to bottom in a fresh kernel.
- Random experiments use explicit seeds where reproducibility matters.
- Floating-point results are checked with tolerances, not exact equality.
- Timing values are generated at runtime and must not be presented as universal benchmarks.
- Notebooks should not contain credentials, private data, or machine-specific absolute paths.
