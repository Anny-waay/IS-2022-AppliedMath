import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

def generate_norm_graph(q):
    data = [norm(q[i] - q[i + 1]) for i in range(len(q) - 1)]
    plt.title("График изменения среднеквадратического отклонения")
    plt.plot(data)
    plt.show()


def numerical_solve(p, q0, eps, max_it_count):
    print(f"Численное моделирование с начальным вектором {q0}")
    it_count = 0
    q = [q0]
    while it_count < max_it_count:
        qn = q0 * p
        q.append(qn)
        q0 = qn
        if norm(q[-1] - q[-2]) < eps:
            break
        it_count += 1

    if norm(q[-1] - q[-2]) > eps:
        print(f"Невозможно достичь точности {eps} за {max_it_count} итераций")

    generate_norm_graph(q)

    print("Ответ")
    print(q0)
    print("\n")

def analitical_solve(p):
    a = p.T
    for i in range(len(a)):
        a[i, i] -= 1

    a[len(a)-1] = [np.ones(len(a))]

    b = np.zeros(len(a))
    b[-1] = 1

    answ = np.linalg.lstsq(a, b, rcond=None)[0]
    print("Аналитическое моделирование")
    print("Ответ")
    print(answ)
    print("\n")

# p = np.matrix([
#     [0.6, 0.2, 0.1, 0.1, 0, 0, 0, 0],
#     [0.5, 0.3, 0, 0, 0.1, 0.1, 0, 0],
#     [0.5, 0, 0.3, 0, 0.1, 0, 0.1, 0],
#     [0.5, 0, 0, 0.3, 0, 0.1, 0.1, 0],
#     [0, 0.4, 0.3, 0, 0.2, 0, 0, 0.1],
#     [0, 0.5, 0, 0.1, 0, 0.2, 0, 0.2],
#     [0, 0, 0.1, 0.2, 0, 0, 0.2, 0.5],
#     [0, 0, 0, 0, 0.2, 0.6, 0.1, 0.1]]
# )

p = np.matrix([
    [0.2, 0, 0.1, 0, 0.3, 0, 0.4, 0],
    [0, 0.5, 0, 0.1, 0, 0.2, 0, 0.2],
    [0.4, 0, 0.3, 0, 0, 0.2, 0, 0.1],
    [0, 0.2, 0, 0.3, 0.4, 0, 0.1, 0],
    [0.5, 0, 0, 0.3, 0.1, 0, 0, 0.1],
    [0, 0.4, 0.1, 0, 0, 0.2, 0.3, 0],
    [0.4, 0, 0, 0.3, 0, 0.1, 0.2, 0],
    [0, 0.1, 0.6, 0, 0.2, 0, 0, 0.1]]
)

q0 = np.matrix([[1, 0, 0, 0, 0, 0, 0, 0]])
numerical_solve(p, q0, 0.001, 10000)

q0 = np.matrix([[0, 0, 1, 0, 0, 0, 0, 0]])
numerical_solve(p, q0, 0.001, 10000)

analitical_solve(p)
