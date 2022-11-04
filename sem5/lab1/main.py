import numpy as np

def check_negative(b):
    for i in range(len(b)):
        if b[i] < 0:
            return i + 1
    return -1

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

def make_basis(simplex_table, delta):
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
        l= list(np.abs(simplex_table[i_neg_b, 1:]))
        j_ind = find_max_el_ind(l, basis_j[prev_base]-1)
        change_basis(simplex_table, i_neg_b, j_ind)
        prev_base = basis_i.index(i_neg_b)
        basis_j[prev_base] = j_ind
        i_neg_b = check_negative(simplex_table[1:, 0])

    replace_neg_zero(simplex_table)
    change_delta(simplex_table, delta, basis_j, basis_i)
    my_round(simplex_table, delta)

def pivot(simplex_table, delta):
    l = list(delta)
    j_base = l.index(max(l))
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

def display_answer(simplex_table, delta):
    answers = np.zeros(len(simplex_table[0]) - 1)
    basis_j, basis_i = find_basis(simplex_table)
    for z in range(len(basis_j)):
        answers[basis_j[z] - 1] = simplex_table[basis_i[z]][0]
    for i in range(len(answers)):
        print("x", i + 1, "=", answers[i])
    print("f =", delta[0])

def solve(simplex_table):
    delta = np.zeros(simplex_table[0].shape)
    make_basis(simplex_table, delta)
    flag = True
    while flag:
        if max(delta[1:]) <= 0:
            flag = False
            my_round(simplex_table, delta)
            display_answer(simplex_table, delta)
        else:
            flag = pivot(simplex_table, delta)
            if not flag:
                print("Область не ограничена")

for i in range(1, 8):
    test_path = "tests/test" + str(i) + ".txt"
    simplex_table = np.loadtxt(test_path, dtype=float)
    solve(simplex_table)
    print()