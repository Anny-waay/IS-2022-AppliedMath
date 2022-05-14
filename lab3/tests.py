import numpy as np
import functions as func
import matrixGenerator as generate
import time

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def getIterationsGraph(matrixSize, JakobiMethodIterations, LUMethodIterations):
    plt.plot(matrixSize, JakobiMethodIterations, label="Jakobi Iterations")
    plt.plot(matrixSize, LUMethodIterations, label="LU Iterations")
    plt.xlabel('Matrix size')
    plt.ylabel('Iterations')
    plt.title('Matrix size - Iterations')
    plt.legend()
    plt.savefig('Graphs/Iterations.jpg')


def getTimeGraph(matrixSize, JakobiMethodTime, LUMethodTime):
    plt.plot(matrixSize, JakobiMethodTime, label="Jakobi time ")
    plt.plot(matrixSize, LUMethodTime, label="LU time")
    plt.xlabel('Matrix size')
    plt.ylabel('Time, sec')
    plt.title('Matrix Size - Time')
    plt.legend()
    plt.savefig('Graphs/Time.jpg')


def getLUTimeGraph(matrixSize, LUBuildTime):
    plt.plot(matrixSize, LUBuildTime, label="LU time")
    plt.xlabel('Matrix size')
    plt.ylabel('Time, sec')
    plt.title('LU decomposition')
    plt.legend()
    plt.savefig('Graphs/LUTimeDecomposition.jpg')


def AKMatrixTask4(n):
    print("diagonal dominance \n\n")
    for k in range(1, n):
        mat = generate.generateAKMatrix(k, k)
        b = generate.generateBcof(mat)
        print("---------------[A]---------------\n")
        print(f"{mat.toarray()} \n")
        print("---------------[B]---------------\n")
        print(f"{b} \n")
        print("---------------[SOL Jakobi]---------------\n")
        print(f"{func.Jakobi(mat, b, 0.1)[0]} \n\n")

def GilbertTask5():
    print("Gilbert \n \n")
    for size in range(1, 4):
        mat = generate.generateGilbert(size)
        b = generate.generateBcof(mat)
        print("---------------[A]---------------\n")
        print(f"{mat.toarray()} \n")
        print("---------------[B]---------------\n")
        print(f"{b} \n")
        print("---------------[SOL Jakobi]---------------\n")
        print(f"{func.Jakobi(mat, b, 0.1)[0]} \n\n")

def CompareLUJacobiTask6():

    matrixSize = [10, 50, 100, 200, 500]


    eps = 0.1

    LUBuildTime = np.array([])
    LUMethodTime = np.array([])
    JakobiMethodTime = np.array([])

    LUMethodIterations = np.array([])
    JakobiMethodIterations = np.array([])

    for n in matrixSize:

        matrix = generate.generateDiagMatrix(n)

        buildLUTime = time.time()
        buildLUTime = time.time() - buildLUTime

        LUBuildTime = np.append(LUBuildTime, buildLUTime)

        bVector = generate.generateRandomBVector(n)

        LUResolveTime = time.time()
        LUResolve = func.solution(matrix, bVector)
        LUResolveTime = time.time() - LUResolveTime

        LUMethodTime = np.append(LUMethodTime, LUResolveTime)
        LUMethodIterations = np.append(LUMethodIterations, LUResolve[1])

        jTime = time.time()
        JakobiResolve = func.Jakobi(matrix, bVector, eps)
        jTime = time.time() - jTime

        JakobiMethodTime = np.append(JakobiMethodTime, jTime)
        JakobiMethodIterations = np.append(
            JakobiMethodIterations, JakobiResolve[1])

        print(f"end {n}")

    getIterationsGraph(matrixSize, JakobiMethodIterations, LUMethodIterations)
    plt.clf()
    getTimeGraph(matrixSize, JakobiMethodTime, LUMethodTime)
    plt.clf()
    getLUTimeGraph(matrixSize, LUBuildTime)

AKMatrixTask4(10)
print("\n \n")
GilbertTask5()
CompareLUJacobiTask6()