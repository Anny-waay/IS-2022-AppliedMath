import scipy.sparse as sps
import numpy as np
from random import choice


def generateDiagMatrix(n):
    matrix = sps.rand(n, n, density=0.1, format='csr', dtype=np.int16)
    matrix.setdiag(1)
    return matrix


def generateRandomBVector(n):
    return np.random.rand(n)


def matrixGenerator(n, getItem):
    matrix = sps.csr_matrix(sps.rand(n, n, density=0.0))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = 1.0/ (i + j + 1.0)
    return matrix


def generateGilbert(n):
    f = lambda i, j: np.float16(1)/ (np.float16(i) +np.float16(j) - np.float16(1.0))

    return matrixGenerator(n, f)


def generateAKMatrix(n, k):
    matrix = sps.rand(n, n, density=0.0, format='csr', dtype=np.float64)
    nums = [ 0, -1, -2, -3, -4 ]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            ranadNum = choice(nums)
            if ranadNum == 0 : continue
            matrix[i, j] += ranadNum + 10 ** (-k)
    for i in range(n):
        sum_ = 0
        for j in range(n):
            if i == j: continue
            sum_ += matrix[i, j]
        matrix[i, i] = sum_

    return matrix


def generateBcof(A):
    n = len(A.indptr) - 1
    y = np.array([])
    for i in range(1, n + 1):
        y = np.append(y, i)

    return np.array(A.dot(y))