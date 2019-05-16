import numpy as np
import numba
from numba import jit, njit, prange, cuda, float32
from sklearn.utils.validation import check_array
from scipy import linalg
from blis.py import einsum

# additional numba speedups
@jit(nopython=True, parallel=True)
def dot_nb(A,B):
    return np.dot(A,B)
# dot_nb = nb.jit(nb.float64[:,:](nb.float64[:,:], nb.float64[:,:]), nopython = True)(dot_py)


def _impose_f_order(X):
    """Helper Function"""
    # important to access flags instead of calling np.isfortran,
    # this catches corner cases.
    if X.flags.c_contiguous:
        return check_array(X.T, copy=False, order='F'), True
    else:
        return check_array(X, copy=False, order='F'), False

def fast_dot(A, B):
    """Compute fast dot products directly calling BLAS.
    This function calls BLAS directly while warranting Fortran contiguity.
    This helps avoiding extra copies `np.dot` would have created.
    For details see section `Linear Algebra on large Arrays`:
    http://wiki.scipy.org/PerformanceTips
    Parameters
    ----------
    A, B: instance of np.ndarray
        input matrices.
    """
    if A.dtype != B.dtype:
        raise ValueError('A and B must be of the same type.')
    if A.dtype not in (np.float32, np.float64):
        raise ValueError('Data must be single or double precision float.')

    dot = linalg.get_blas_funcs('gemm', (A, B))
    A, trans_a = _impose_f_order(A)
    B, trans_b = _impose_f_order(B)
    return dot(alpha=1.0, a=A, b=B, trans_a=trans_a, trans_b=trans_b)

# def matmul(A,B):
#     blas = Blas()
#     N = A.shape[0]
#     D = np.zeros_like(A, order='F')
#     blas.gemm('T', 'T', N, N, N, 1.0, A, B, 1.0, D)
#     return D

@njit(parallel=True)
def matmul(matrix1,matrix2):
    rmatrix = np.zeros_like(matrix1)
    for i in prange(len(matrix1)):
        for j in prange(len(matrix2[0])):
            for k in prange(len(matrix2)):
                rmatrix[i][j] += matrix1[i][k] * matrix2[k][j]
    return rmatrix

#Controls threads per block and shared memory usage.
#The computation will be done on blocks of TPBxTPB elements.
TPB = 16

@cuda.jit
def fast_matmul(A, B, C):
    """
    Perform matrix multiplication of C = A * B
    Each thread computes one element of the result matrix C
    """

    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(int(A.shape[1] / TPB)):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp
