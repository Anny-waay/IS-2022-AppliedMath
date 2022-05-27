import scipy.sparse as sps
import numpy as np
from random import choice


def generateSimMatrix(n):
    matrix = sps.rand(n, n, density=0.6, format='csr', dtype=np.int8)
    for i in range(1, n):
        for j in range(i):
            matrix[i, j] = matrix[j, i]
    return matrix


def generateGilbert(n):
    matrix = sps.csr_matrix(sps.rand(n, n, density=0.0))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = 1.0 / (i + j + 1.0)
    return matrix


def generateAKMatrix(n):
    nums = [0, -1, -2, -3, -4, -5, -6]
    noise = 10 ** (-n)
    matrix = sps.rand(n, n, density=0.0, format='csr', dtype=np.float64)

    for i in range(n):
        for j in range(n):
            matrix[i, j] = choice(nums)

    for i in range(n):
        matrix[i, i] = -(sum(sum(matrix[i].toarray())) - matrix[i, i]) + noise

    for i in range(1, n):
        for j in range(i):
            matrix[i, j] = matrix[j, i]

    return matrix
