import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from sklearn.svm import NuSVC
from sklearn.model_selection import train_test_split
import warnings

warnings.simplefilter(action='ignore')

def classify(x, y):
    if x ** 2 + y ** 2 < 0.25:
        return 1
    else:
        return 0


def generate_data(n):
    data = np.zeros((n, 3))
    for i in range(n):
        data[i][0] = random.uniform(-1.0, 1.0)
        data[i][1] = random.uniform(-1.0, 1.0)
        data[i][2] = classify(data[i][0], data[i][1])
    return pd.DataFrame(data, columns=['x', 'y', 'inside'])


def accuracy(y_pred, y_test):
    T = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            T += 1
    return T/len(y_pred)


def precision(y_pred, y_test):
  TP = 0
  FP = 0
  for i in range(len(y_pred)):
    if y_pred[i] == y_test[i] == 1:
      TP += 1
    if y_pred[i] == 1 and y_pred[i] != y_test[i]:
      FP += 1
  return TP / (TP + FP)


def recall(y_pred, y_test):
  TP = 0
  FN = 0
  for i in range(len(y_pred)):
    if y_pred[i] == y_test[i] == 1:
      TP += 1
    if y_pred[i] == 0 and y_pred[i] != y_test[i]:
      FN += 1
  return TP / (TP + FN)


def F1(y_pred, y_test):
  TP = 0
  FP = 0
  FN = 0
  for i in range(len(y_pred)):
    if y_pred[i] == y_test[i] == 1:
      TP += 1
    if y_pred[i] == 1 and y_pred[i] != y_test[i]:
      FP += 1
    if y_pred[i] == 0 and y_pred[i] != y_test[i]:
      FN += 1
  return TP / (TP + 0.5 * (FP + FN))


def graphic(data, pred, func, train_n, nu, kernel):
    plt.figure(figsize=(10, 10))

    xx, yy = np.meshgrid(np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000))
    Z = func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 10))

    plt.contour(xx, yy, Z, levels=[0], linewidths=2)

    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=pred, cmap='Set3')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Train data volume={train_n} nu={nu} kernel={kernel}")
    plt.show()


def report(kernel, n, nu, with_graphic):
    data = generate_data(n)

    X = data[['x', 'y']]
    y = data['inside']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y)

    nuSVC = NuSVC(nu=nu, kernel=kernel)
    nuSVC.fit(X_train, y_train)
    y_pred = nuSVC.predict(X_test)

    if with_graphic:
        graphic(X_test, y_pred, nuSVC.decision_function, 0.7 * n, nu, kernel)

    print(pd.DataFrame([[accuracy(y_pred, y_test.values.astype(np.int64)),
          precision(y_pred, y_test.values.astype(np.int64)),
          recall(y_pred, y_test.values.astype(np.int64)),
          F1(y_pred, y_test.values.astype(np.int64))]],
          columns=['accuracy', 'precision', 'recall', 'F1-мера'], index=[f'{kernel} {int(0.7 * n)} {nu}']), "\n")

def choose_kernel():
    print("Kernel choice\n")
    report('linear', 5000, 0.1, True)
    report('poly', 5000, 0.1, True)
    report('rbf', 5000, 0.1, True)
    report('sigmoid', 5000, 0.1, True)

def choose_train_data_volume():
    print("Train data volume choice\n")
    for n in [1000, 3000, 5000, 7000, 10000, 12000, 15000]:
        report('rbf', n, 0.1, False)

def choose_nu():
    print("Nu choice\n")
    for nu in [0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01]:
        report('rbf', 5000, nu, False)

choose_kernel()
choose_train_data_volume()
choose_nu()