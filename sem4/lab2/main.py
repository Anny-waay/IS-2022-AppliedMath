import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 15)
mpl.rcParams["axes.unicode_minus"] = False


def func1(x, y):
    return np.cos(x) + y ** 2


def df1_x(x, y):
    return -np.sin(x)


def df1_y(x, y):
    return 2 * y


def func2(x, y):
    return x ** 2 + y ** 2 - x * y


def df2_x(x, y):
    return 2 * x - y


def df2_y(x, y):
    return 2 * y - x


def func3(x, y):
    return x ** 2 + 1/2 * y ** 2 - y * x + 4 * y


def df3_x(x, y):
    return 2 * x - y


def df3_y(x, y):
    return y - x + 4


def make_graph(title, a, b, f, x_list, y_list):
    X = np.arange(a, b, 0.1)
    Y = np.arange(a, b, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([X.ravel(), Y.ravel()]).T
    Z = f(Z[:, 0], Z[:, 1])
    Z = Z.reshape(X.shape)

    m = plt.contour(X, Y, Z, 40)
    plt.title(title)
    plt.colorbar(m)
    plt.plot(x_list, y_list, 'o-', c="purple")
    plt.show()


def golden_ratio(a, b, eps, x, y, f, df_x, df_y):
    phi = (1 + 5 ** 0.5) / 2
    l1 = b - (b - a)/phi
    l2 = a + (b - a)/phi
    f1 = f(x-l1*df_x(x,y), y-l1*df_y(x,y))
    f2 = f(x-l2*df_x(x,y), y-l2*df_y(x,y))
    while (b - a) > eps:
        if f1 < f2:
            b, l2, f2 = l2, l1, f1
            l1 = b - (b - a)/phi
            f1 = f(x-l1*df_x(x,y), y-l1*df_y(x,y))
        else:
            a, l1, f1 = l1, l2, f2
            l2 = a + (b - a)/phi
            f2 = f(x-l2*df_x(x,y), y-l2*df_y(x,y))

    return (a + b) / 2


def fibonacci_num(a, b, eps, x, y, f, df_x, df_y):
    fib_n_2 = (b-a) / eps
    fib_list = []
    fib_list.append(0)
    fib_list.append(1)
    s = 1
    while fib_n_2 > fib_list[s]:
        fib_list.append(fib_list[s - 1] + fib_list[s - 2])
        s += 1
    n = s - 2
    l1 = b - ((float)(fib_list[n+1])/fib_list[n+2]) * (b-a)
    l2 = a + ((float)(fib_list[n+1])/fib_list[n+2]) * (b-a)
    f1 = f(x-l1*df_x(x,y), y-l1*df_y(x,y))
    f2 = f(x-l2*df_x(x,y), y-l2*df_y(x,y))
    i = n-1
    while i >= 2:
        if f1 < f2:
            b, l2, f2 = l2, l1, f1
            l1 = a + ((float)(fib_list[i]) / fib_list[i+2]) * (b-a)
            f1 = f(x - l1 * df_x(x, y), y - l1 * df_y(x, y))
        else:
            a, l1, f1 = l1, l2, f2
            l2 = a + ((float)(fib_list[i+1]) / fib_list[i+2]) * (b-a)
            f2 = f(x-l2*df_x(x,y), y-l2*df_y(x,y))
        i -= 1

    return (a + b) / 2


def grad_down_const_step(title, f, df_x, df_y,  x0, y0, eps, alpha, a, b):
    x_list, y_list = [], []

    x_list.append(x0)
    y_list.append(y0)

    last_x, last_y = x0, y0
    curr_x = last_x - alpha * df_x(last_x, last_y)
    curr_y = last_y - alpha * df_y(last_x, last_y)
    counter = 1
    while(abs(f(curr_x, curr_y)-f(last_x, last_y)) > eps):
        x_list.append(curr_x)
        y_list.append(curr_y)

        last_x, last_y = curr_x, curr_y
        curr_x = last_x - alpha * df_x(last_x, last_y)
        curr_y = last_y - alpha * df_y(last_x, last_y)
        counter += 1

    title = title + "\nConst Step"
    make_graph(title, a, b, f, x_list, y_list)

    print("Constant step gradient descent")
    print("Number of iterations:", counter)
    print("x:", curr_x)
    print("y:", curr_y, "\n")


def grad_down_crushed_step(title,f, df_x, df_y,  x0, y0, eps, alpha, a, b):
    x_list, y_list= [], []

    x_list.append(x0)
    y_list.append(y0)

    last_x, last_y = x0, y0
    curr_x = last_x - alpha * df_x(last_x, last_y)
    curr_y = last_y - alpha * df_y(last_x, last_y)
    counter = 1
    while(abs(f(curr_x, curr_y)-f(last_x, last_y)) > eps):
        x_list.append(curr_x)
        y_list.append(curr_y)

        last_x, last_y = curr_x, curr_y
        if(f(curr_x, curr_y)>f(last_x, last_y)):
            alpha *= 0.5
        curr_x = last_x - alpha * df_x(last_x, last_y)
        curr_y = last_y - alpha * df_y(last_x, last_y)
        counter += 1

    title = title + "\nCrushed Step"
    make_graph(title, a, b, f, x_list, y_list)

    print("Gradient descent with crushed step")
    print("Number of iterations:", counter)
    print("x:", curr_x)
    print("y:", curr_y, "\n")


def grad_down_gold(title, f, df_x, df_y,  x0, y0, eps, a, b):
    x_list, y_list = [], []

    x_list.append(x0)
    y_list.append(y0)

    last_x, last_y = x0, y0
    alpha = golden_ratio(a, b, eps, last_x, last_y, f, df_x, df_y)
    curr_x = last_x - alpha * df_x(last_x, last_y)
    curr_y = last_y - alpha * df_y(last_x, last_y)
    counter = 1
    while abs(f(curr_x, curr_y)-f(last_x, last_y)) > eps:
        x_list.append(curr_x)
        y_list.append(curr_y)

        last_x, last_y = curr_x, curr_y
        alpha = golden_ratio(a, b, eps, last_x, last_y, f, df_x, df_y)
        curr_x = last_x - alpha * df_x(last_x, last_y)
        curr_y = last_y - alpha * df_y(last_x, last_y)
        counter += 1

    title = title + "\nGold"
    make_graph(title, a, b, f, x_list, y_list)

    print("Gradient descent with golden ratio")
    print("Number of iterations:", counter)
    print("x:", curr_x)
    print("y:", curr_y, "\n")


def grad_down_fibonacci(title, f, df_x, df_y,  x0, y0, eps, a, b):
    x_list, y_list = [], []

    x_list.append(x0)
    y_list.append(y0)

    last_x, last_y = x0, y0
    alpha = fibonacci_num(a, b, eps, last_x, last_y, f, df_x, df_y)
    curr_x = last_x - alpha * df_x(last_x, last_y)
    curr_y = last_y - alpha * df_y(last_x, last_y)
    counter = 1
    while abs(f(curr_x, curr_y)-f(last_x, last_y)) > eps:
        x_list.append(curr_x)
        y_list.append(curr_y)

        last_x, last_y = curr_x, curr_y
        alpha = fibonacci_num(a, b, eps, last_x, last_y, f, df_x, df_y)
        curr_x = last_x - alpha * df_x(last_x, last_y)
        curr_y = last_y - alpha * df_y(last_x, last_y)
        counter += 1

    title = title + "\nFibonacci"
    make_graph(title, a, b, f, x_list, y_list)

    print("Gradient descent with fibonacci numbers")
    print("Number of iterations:", counter)
    print("x:", curr_x)
    print("y:", curr_y, "\n")


def conjugate_grad(title, f, a, b, matrix, free_vector, xy, eps):
    x_list, y_list = [], []
    x_list.append(xy[0])
    y_list.append(xy[1])
    grad = np.dot(matrix, xy) + free_vector
    pk = -grad
    xy_k = xy
    grad_norm = np.amax(np.abs(grad))
    counter = 0
    while grad_norm > eps:
        alpha_k = - np.dot(grad, pk) / np.dot(np.dot(pk, matrix.T), pk)
        xy_k = xy_k + alpha_k * pk
        grad_k = np.dot(matrix, xy_k) + free_vector
        beta_k = max(0, np.dot(grad_k, grad_k)/np.dot(grad, grad))
        pk = -grad_k + beta_k * pk
        grad = grad_k
        grad_norm = np.amax(abs(grad))
        x_list.append(xy_k[0])
        y_list.append(xy_k[1])
        counter += 1

    title = title + "\nConjugate gradient"
    make_graph(title, a, b, f, x_list, y_list)

    print("Conjugate gradient")
    print("Number of iterations:", counter)
    print("x:", xy_k[0])
    print("y:", xy_k[1], "\n")


title = "Function: cos(x) + y^2"
print(title)
grad_down_const_step(title, func1, df1_x, df1_y, 2, 4, 0.00001, 0.3, -5, 5)
grad_down_crushed_step(title, func1, df1_x, df1_y, 2, 4, 0.00001, 0.5, -5, 5)
grad_down_gold(title, func1, df1_x, df1_y, 2, 4, 0.00001, -5, 5)
grad_down_fibonacci(title, func1, df1_x, df1_y, 2, 4, 0.00001, -5, 5)

title = "Function: x^2 + y^2 - xy"
print(title)
grad_down_const_step(title, func2, df2_x, df2_y, 5, 8, 0.00001, 0.3, -10, 10)
grad_down_crushed_step(title, func2, df2_x, df2_y, 5, 8, 0.00001, 0.5, -10, 10)
grad_down_gold(title, func2, df2_x, df2_y, 5, 8, 0.00001, -10, 10)
grad_down_fibonacci(title, func2, df2_x, df2_y, 5, 8, 0.00001, -10, 10)

conjugate_grad(title, func2, -10, 10, np.array([[2, -1], [-1, 2]]), [0, 0], [5, 8], 0.00001)

title = "Function:  x^2 + 1/2*y^2 - yx + 4y"
print(title)
grad_down_const_step(title, func3, df3_x, df3_y, -15, 15, 0.00001, 0.5, -18, 18)
grad_down_crushed_step(title, func3, df3_x, df3_y, -15, 15, 0.00001, 0.6, -18, 18)
grad_down_gold(title, func3, df3_x, df3_y, -15, 15, 0.00001, -18, 18)
grad_down_fibonacci(title, func3, df3_x, df3_y, -15, 15, 0.00001, -18, 18)

conjugate_grad(title, func3, -18, 18, np.array([[2, -1], [-1, 1]]), [0, 4], [-15, 15], 0.00001)
