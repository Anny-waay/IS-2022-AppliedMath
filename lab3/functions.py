import scipy.sparse as sps
import scipy.stats as stats
import numpy as np
from numpy import linalg
from numpy import array, zeros, diag, diagflat, dot


def LU(A):
    n = len(A.indptr) - 1

    U = sps.csr_matrix(sps.rand(n, n, density=0.0))
    L = sps.csr_matrix(sps.rand(n, n, density=0.0))

    for i in range(n):
        for j in range(n):
            U[0, i] = A[0, i]
            L[i, 0] = A[i, 0] / U[0, 0]

            s = 0.0

            for k in range(i):
                s += L[i, k] * U[k, j]
            U[i, j] = A[i, j] - s

            if i > j:
                L[j, i] = 0
            else:
                s = 0.0
                for k in range(i):
                    s += L[j, k] * U[k, i]
                L[j, i] = (A[j, i] - s) / U[i, i]

    return [L, U]


def inverseMatrix(L, U):
    n = len(L.indptr) - 1

    eMatrix = sps.csr_matrix(sps.rand(n, n, density=0.0))
    eMatrix.setdiag(1)

    y = np.array([])
    resultMatrix = [np.array([])]

    iterCounter = 0

    for k in range(0, n):
        e = eMatrix.getrow(k).toarray()[0]
        temp = np.array([])
        for i in range(0, n):
            sum = 0
            for p in range(0, i):
                sum += L[i, p] * temp[p]
            yi = e[i] - sum
            temp = np.append(temp, yi)
            iterCounter += 1
        y = np.append(y, temp)
        iterCounter += 1

    y = y.reshape(n, n)

    for k in range(0, n):
        yi = y[k]
        x = np.zeros(n)
        for i in range(0, n):
            sum = 0
            for k in range(0, i):
                sum += U[n - i - 1, n - k - 1] * x[n - k - 1]
            x[n - i - 1] = 1/U[n - i - 1, n - i - 1] * (yi[n - i - 1] - sum)
            iterCounter += 1
        resultMatrix = np.append(resultMatrix, x)
        iterCounter += 1

    resultMatrix = resultMatrix.reshape(n, n)
    resultMatrix = resultMatrix.transpose()

    return (resultMatrix, iterCounter)


def Jakobi(A, B, eps):
    # A - матрица коэффициентов
    # В - столбец свободных членов

    maxIter = 1000
    norm = 10  # норма, определяемая как наибольшая разность компонент столбца иксов соседних итераций

    n = len(A.indptr) - 1  # размерность матрицы
    x2 = B.copy()

    X = np.array([])  # начальное приближение
    ab = A.copy()
    ab = ab.multiply(sps.csr_matrix(B))  # A*B
    iterNum = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                X = np.append(X, ab[i, j])
    while abs(norm) > eps and iterNum < maxIter:
        for i in range(n):
            x2[i] = B[i]
            for j in range(n):
                if i != j:
                    x2[i] = x2[i] - (A[i, j] * X[j])
            x2[i] = x2[i]/A[i, i]
        norm = X[0] - x2[0]
        for i in range(n):
            if (abs(X[i] - x2[i]) > norm):
                norm = abs(X[i] - x2[i])
            X[i] = x2[i]
        iterNum += 1
    return (X, iterNum)


def solution(A, b):
    luMatrix = LU(A)
    L = luMatrix[0]
    U = luMatrix[1]

    n = len(U.indptr) - 1

    y = np.array([])

    iter_ = 0

    for i in range(0, n):
        s = 0
        for k in range(0, i):
            s += y[k] * L[i, k]
        y = np.append(y, b[i] - s)
        iter_ += 1

    x = np.zeros(n)

    for i in range(0, n):
        s = 0

        for k in range(0, i):
            s += U[n - i - 1, n - k - 1] * x[n - k - 1]

        x[n - i - 1] = 1/U[n - i - 1, n - i - 1] * (y[n - i - 1] - s)
        iter_ += 1
    return (x, iter_)