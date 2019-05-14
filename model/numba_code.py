import numpy as np
import numba
from numba import jit

# additional numba speedups
@jit(nopython=True)
def dot_np(A,B):

    C = np.dot(A,B)
    return C

@jit('void(float64[:,:],float64[:,:],float64[:,:])')
def matmul(matrix1,matrix2,rmatrix):
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                rmatrix[i][j] += matrix1[i][k] * matrix2[k][j]
    return rmatrix
