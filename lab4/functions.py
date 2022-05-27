import scipy.sparse as sps
from numpy import diag
from math import atan, cos, sin, pi


def maxElem(A, n):
    aMax, k, l = 0.0, 0, 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if abs(A[i, j]) >= aMax:
                aMax = abs(A[i, j])
                k = i
                l = j
    return k, l


def makeEmatrix(n):
    eMatrix = sps.csr_matrix(sps.rand(n, n, density=0.0))
    eMatrix.setdiag(1)
    return eMatrix


def rotate(A, n, k, l):
    U = makeEmatrix(n)
    if A[k, k] != A[l, l]:
        phi = 0.5 * atan((2 * A[k, l]) / (A[k, k] - A[l, l]))
    else:
        phi = pi / 4
    phi = phi
    U[k, k], U[l, l] = cos(phi), cos(phi)
    U[k, l], U[l, k] = -sin(phi), sin(phi)
    U_transp = U.transpose()

    A1 = (U_transp.dot(A)).dot(U)
    return A1, U


def norm(A, n):
    sum = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            sum += A[i, j] ** 2
    return sum ** 0.5


def jacobi(A, n, eps):
    U = makeEmatrix(n)
    while norm(A, n) > eps:
        k, l = maxElem(A, n)
        A1, Utemp = rotate(A, n, k, l)
        A = A1
        U *= Utemp
    return diag(A.toarray()), U


