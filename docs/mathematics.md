# Mathematical Foundations

This document explains the operations exposed by the Matrix class. The implementation uses NumPy's numerical routines; the formulas below describe the mathematics, not necessarily the algorithm used internally.

## 1. Matrix dimensions

A matrix A with m rows and n columns has shape m × n. The shape determines which operations are valid. Addition and subtraction require equal shapes. Matrix multiplication of A (m × n) and B (n × p) produces C (m × p).

## 2. Addition and subtraction

For equal-shaped matrices:

```
(A + B)[i,j] = A[i,j] + B[i,j]
(A - B)[i,j] = A[i,j] - B[i,j]
```

## 3. Element-wise and scalar multiplication

Element-wise multiplication, also called the Hadamard product, multiplies corresponding entries:

```
(A ⊙ B)[i,j] = A[i,j] B[i,j]
```

Scalar multiplication multiplies every entry by the same scalar. In this library, `A * B` is element-wise multiplication and `A @ B` is matrix multiplication.

## 4. Matrix multiplication

If A has shape m × n and B has shape n × p, then:

```
C[i,j] = sum(A[i,k] * B[k,j] for k in range(n))
```

The output shape is m × p. Dense classical multiplication has arithmetic complexity O(mnp); actual performance depends on NumPy's optimized numerical libraries and the hardware.

## 5. Transpose

The transpose swaps rows and columns: `A.T[i,j] = A[j,i]`. A matrix with shape m × n becomes one with shape n × m.

## 6. Determinant and inverse

The determinant is defined for square matrices. A matrix is invertible exactly when its determinant is nonzero in exact arithmetic. The inverse satisfies:

```
A @ inverse(A) = inverse(A) @ A = I
```

In floating-point arithmetic, a computed determinant close to zero is not a robust standalone test of singularity. Use a solver or singular values and consider conditioning.

## 7. Rank and pseudoinverse

The rank is the dimension of the column space and equals the dimension of the row space. Numerical rank depends on a tolerance because floating-point singular values are rarely exactly zero.

The Moore–Penrose pseudoinverse A⁺ generalizes the inverse to rectangular and rank-deficient matrices. It can produce a least-squares solution and, among those solutions, a minimum-norm solution under standard conditions.

## 8. Trace and norms

For square A, trace(A) is the sum of the diagonal entries. The default norm in this library uses NumPy's matrix-norm convention with `ord=None`, which is the Frobenius norm:

```
||A||_F = sqrt(sum_i sum_j |A[i,j]|²)
```

Other supported norm orders follow NumPy's `numpy.linalg.norm` behavior for matrices.

## 9. Condition number

The condition number measures sensitivity to small perturbations. In the 2-norm, for a nonsingular matrix:

```
cond₂(A) = largest_singular_value(A) / smallest_singular_value(A)
```

A large condition number means that small changes in the input may cause large changes in the solution. A small residual alone does not guarantee a small forward error.

## 10. Eigenvalues and eigenvectors

An eigenpair satisfies:

```
A v = λ v, where v is nonzero
```

Eigenvalues describe invariant scaling factors along eigenvector directions. Real matrices can have complex eigenvalues and eigenvectors. The library returns eigenvectors as columns in the eigenvector matrix.

## 11. Solving Ax = b

The solve method uses NumPy's direct solver and requires square A. For a unique solution, A must be nonsingular. The residual is:

```
r = A x - b
```

Its norm measures how closely the computed solution satisfies the equations. To assess solution accuracy, also consider conditioning and, when available, compare with a known reference solution.

## 12. Floating-point behavior

Results use float64 storage. Rounding means algebraic identities generally hold only up to a tolerance. Tests should normally use `numpy.testing.assert_allclose` instead of exact equality for floating-point values.

## Further reading

- NumPy linear algebra: https://numpy.org/doc/stable/reference/routines.linalg.html
- NumPy array basics: https://numpy.org/doc/stable/user/basics.html
- SciPy linear algebra: https://docs.scipy.org/doc/scipy/reference/linalg.html
