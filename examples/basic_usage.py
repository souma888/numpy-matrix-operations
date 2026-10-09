from matrixlib import Matrix

a = Matrix([[1, 2], [3, 4]])
b = Matrix([[5, 6], [7, 8]])

print("A:")
print(a)
print("\nB:")
print(b)
print("\nA + B:")
print(a + b)
print("\nA @ B (matrix multiplication):")
print(a @ b)
print("\nTranspose of A:")
print(a.T)
print("\nDeterminant of A:", a.determinant())
print("\nRank of A:", a.rank())
