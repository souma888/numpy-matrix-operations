"""Demonstrate the main public operations in the Matrix class.

Run from the repository root after installing with: python -m pip install -e .
"""
import numpy as np

from matrixlib import Matrix

a = Matrix([[4, 7], [2, 6]])
b = Matrix([[1, 2], [3, 4]])

print("A:")
print(a)
print("\nB:")
print(b)
print("\nA + B:")
print(a + b)
print("\nA - B:")
print(a - b)
print("\nA * B (element-wise):")
print(a * b)
print("\nA @ B (matrix multiplication):")
print(a @ b)
print("\nTranspose of A:")
print(a.T)
print("\nDeterminant of A:", a.determinant())
print("\nInverse of A:")
print(a.inverse())
print("\nPseudoinverse of A:")
print(a.pseudoinverse())
print("\nRank:", a.rank())
print("Trace:", a.trace())
print("Frobenius norm:", a.norm())
print("Condition number:", a.condition_number())
print("Eigenvalues:", a.eigenvalues())

rhs = np.array([1.0, 0.0])
solution = a.solve(rhs)
print("\nSolve Ax = b with b =", rhs)
print("x =", solution)
print("residual Ax - b =", a.data @ solution - rhs)
