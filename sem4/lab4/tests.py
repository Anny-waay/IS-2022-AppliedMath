import functions as func
import matrixGenerator as generate

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')


def simpleTest(n):
    A = generate.generateSimMatrix(n)
    print("---------------[A]---------------\n")
    print(f"{A.toarray()} \n")
    eigenvalues, U = func.jacobi(A, n, 5e-5)
    print("---------------[Eigenvalues Jakobi]---------------\n")
    print(f"{eigenvalues}\n")
    print("---------------[Eigenvectors Jakobi]---------------\n")
    print(f"{U.toarray()}\n")


def AKMatrixTask3(n):
    print("\ndiagonal dominance \n")
    A = generate.generateAKMatrix(n)
    print("---------------[A]---------------\n")
    print(f"{A.toarray()} \n")
    eigenvalues, U = func.jacobi(A, n, 5e-5)
    print("---------------[Eigenvalues Jakobi]---------------\n")
    print(f"{eigenvalues}\n")
    print("---------------[Eigenvectors Jakobi]---------------\n")
    print(f"{U.toarray()}\n")


def GilbertTask(n):
    print("\nGilbert \n")
    A = generate.generateGilbert(n)
    print("---------------[A]---------------\n")
    print(f"{A.toarray()} \n")
    eigenvalues, U = func.jacobi(A, n, 5e-5)
    print("---------------[Eigenvalues Jakobi]---------------\n")
    print(f"{eigenvalues}\n")
    print("---------------[Eigenvectors Jakobi]---------------\n")
    print(f"{U.toarray()}\n")


simpleTest(4)
GilbertTask(4)
AKMatrixTask3(4)