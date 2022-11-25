import numpy as np

def add_variables(simplex_table, signs):
    for i in range(1, len(signs)):
        if signs[i] != "=":
            edit = np.zeros((len(simplex_table), 1))
            simplex_table = np.append(simplex_table, edit, axis=1)
            if "<" in signs[i]:
                simplex_table[i][-1] = 1
            if ">" in signs[i]:
                # simplex_table[i] *= -1
                simplex_table[i][-1] = -1
            if np.all(simplex_table[:, -1] == edit):
                print("Некорректный знак в строке", i+1)
    replace_neg_zero(simplex_table)
    return simplex_table

def check_negative(b):
    min_b = 0
    index = -1
    for i in range(len(b)):
        if b[i] < 0 and b[i] < min_b:
            min_b = b[i]
            index = i + 1
    return index

def find_max_el_ind(b, escape_ind):
    max = 0
    max_i = -1
    for i in range(len(b)):
        if i != escape_ind and b[i] >= max:
            max = b[i]
            max_i = i
    return max_i + 1

def replace_neg_zero(simplex_table):
    for i in range(len(simplex_table)):
        for j in range(len(simplex_table[i])):
            if simplex_table[i][j] == -0.0:
                simplex_table[i][j] = 0.0

def change_delta(simplex_table, delta, basis_j, basis_i):
    for i in range(len(delta)):
        delta[i] = - simplex_table[0][i]
        for z in range(len(basis_j)):
            delta[i] += simplex_table[0][basis_j[z]] * simplex_table[basis_i[z]][i]

def my_round(simplex_table, delta):
    for i in range(len(simplex_table)):
        delta[i] = round(delta[i], 3)
        for j in range(len(simplex_table[i])):
            simplex_table[i][j] = round(simplex_table[i][j], 3)

def find_basis(simplex_table):
    basis_j = []
    basis_i = []
    for j in range(1, len(simplex_table[0])):
        zeroes = simplex_table[1:, j].tolist().count(0)
        ones = simplex_table[1:, j].tolist().count(1)
        if ones == 1 and zeroes == len(simplex_table[1:, j]) - 1:
            one_ind = simplex_table[1:, j].tolist().index(1) + 1
            if one_ind not in basis_i:
                basis_j.append(j)
                basis_i.append(one_ind)
        if len(basis_j) == len(simplex_table) - 1:
            break
    return basis_j, basis_i

def change_basis(simplex_table, i_base, j_base):
    r = simplex_table[i_base][j_base]
    simplex_table[i_base] /= r
    for i in range(1, len(simplex_table)):
        if i != i_base:
            r = simplex_table[i][j_base]
            simplex_table[i] -= r * simplex_table[i_base]

def make_basis(simplex_table, signs):
    simplex_table = add_variables(simplex_table, signs)
    delta = np.zeros(simplex_table[0].shape)
    basis_j, basis_i = find_basis(simplex_table)
    if len(basis_j) != len(simplex_table) - 1:
        for j in range(1, len(simplex_table[0])):
            if j not in basis_j:
                i_base = 1
                while i_base in basis_i:
                    i_base += 1
                if simplex_table[i_base][j] != 0:
                    basis_j.append(j)
                    basis_i.append(i_base)
                    change_basis(simplex_table, i_base, j)
            if len(basis_j) == len(simplex_table) - 1:
                break

    replace_neg_zero(simplex_table)
    i_neg_b = check_negative(simplex_table[1:, 0])
    while i_neg_b != -1:
        prev_base = basis_i.index(i_neg_b)
        if check_negative(simplex_table[i_neg_b, 1:]) == -1:
            print("Решение задачи не существует")
            return -1
        l= list(np.abs(simplex_table[i_neg_b, 1:]))
        j_ind = find_max_el_ind(l, basis_j[prev_base]-1)
        change_basis(simplex_table, i_neg_b, j_ind)
        prev_base = basis_i.index(i_neg_b)
        basis_j[prev_base] = j_ind
        i_neg_b = check_negative(simplex_table[1:, 0])

    replace_neg_zero(simplex_table)
    change_delta(simplex_table, delta, basis_j, basis_i)
    my_round(simplex_table, delta)
    return simplex_table, delta

def pivot(simplex_table, delta, min_max):
    l = list(delta)
    if min_max == "min":
        j_base = l.index(max(l))
    else:
        j_base = l.index(min(l))
    i_base = -1
    for i in range(1, len(simplex_table)):
        if simplex_table[i][j_base] > 0:
            i_base = i
            break
    if i_base == -1:
        return False
    for i in range(i_base + 1, len(simplex_table)):
        if simplex_table[i][j_base] > 0 and (simplex_table[i][0] / simplex_table[i][j_base]) < (simplex_table[i_base][0] / simplex_table[i_base][j_base]):
            i_base = i
    change_basis(simplex_table, i_base, j_base)

    basis_j, basis_i = find_basis(simplex_table)
    change_delta(simplex_table, delta, basis_j, basis_i)
    return True

def display_answer(simplex_table, delta, x_num):
    answers = np.zeros(len(simplex_table[0]) - 1)
    basis_j, basis_i = find_basis(simplex_table)
    for z in range(len(basis_j)):
        answers[basis_j[z] - 1] = simplex_table[basis_i[z]][0]

    print("f =", delta[0], "\n")
    for i in range(len(answers)):
        if i == x_num:
            print("\nДополнительные переменные")
        print("x", i + 1, "=", answers[i])

def solve(simplex_table, signs, min_max):
    x_num = len(simplex_table[0]) - 1
    if make_basis(simplex_table, signs) == -1:
        return
    simplex_table, delta = make_basis(simplex_table, signs)
    flag = True
    while flag:
        if min_max == "min":
            if max(delta[1:]) <= 0:
                flag = False
                my_round(simplex_table, delta)
                display_answer(simplex_table, delta, x_num)
            else:
                flag = pivot(simplex_table, delta, min_max)
                if not flag:
                    print("Область не ограничена")
        elif min_max == "max":
            if min(delta[1:]) >= 0:
                flag = False
                my_round(simplex_table, delta)
                display_answer(simplex_table, delta, x_num)
            else:
                flag = pivot(simplex_table, delta, min_max)
                if not flag:
                    print("Область не ограничена")
        else:
            flag = False
            print("Введите min или max")


# lab1
# for i in range(1, 8):
#     print("Test", i)
#     test_path = "tests-lab1/test" + str(i) + ".txt"
#     data = np.loadtxt(test_path, dtype=str)
#     simplex_table = data[:, :-1].astype("float")
#     signs = list(data[:, -1])
#     min_max = data[0, -1]
#     solve(simplex_table, signs, min_max)
#     print()

data = np.loadtxt("tasks-lab2/task8_2.txt", dtype=str)
simplex_table = data[:, :-1].astype("float")
signs = list(data[:, -1])
min_max = data[0, -1]
solve(simplex_table, signs, min_max)