# Mathematical Foundations

## Addition and subtraction

For matrices of equal shape, operations are performed element by element:

\[
(A+B)_{ij}=A_{ij}+B_{ij}, \qquad
(A-B)_{ij}=A_{ij}-B_{ij}.
\]

## Element-wise multiplication

\[
(A\odot B)_{ij}=A_{ij}B_{ij}.
\]

This differs from matrix multiplication.

## Matrix multiplication

If \(A\) has shape \(m\times n\) and \(B\) has shape \(n\times p\), then \(C=AB\) has shape \(m\times p\):

\[
C_{ij}=\sum_{k=1}^{n}A_{ik}B_{kj}.
\]

The standard dense multiplication cost is \(O(mnp)\).

## Transpose

\[
(A^T)_{ij}=A_{ji}.
\]

## Determinant and inverse

The determinant is defined for square matrices. A matrix has an inverse exactly when it is nonsingular; in exact arithmetic this is equivalent to a nonzero determinant. Floating-point calculations require numerical care, so do not use a tiny computed determinant as the sole test for invertibility.

## Numerical notes

NumPy uses floating-point arithmetic here. Results may contain small rounding errors. Use tolerances in tests (for example, `numpy.testing.assert_allclose`) instead of exact equality for most computed floating-point results.
