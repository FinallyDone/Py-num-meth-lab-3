##################################################################################
#   48 Группа, Абдюшев Тимур, Лабораторная работа 3
#
#   Решение систем нелинейных уравнений. Метод итераций. Метод Зейделя.
#   Метод скорейшего спуска. Метод Ньютона.
#
#   1) Решить систему нелинейных уравнений f(x) = 0 методом простых итераций с точностью 1e-3
#
#   2) Уточнить полученное решение методом Ньютона с точностью 1e-6
#
#   3) Проверка выполнения достаточного условия сходимости
#
##################################################################################

import numpy as np
import math


# Функция печати в консоль матрицы
def print_matrix(matrix, str='', before=8, after=4):
    # Печать числа с настройкой чисел до и после точки
    f = f'{{: {before}.{after}f}}'
    print(str)
    print('\n'.join([f''.join(f.format(el)
                              for el in row)
                     for row in matrix]) + '\n')


# Система нелинейных уравнений
def func(params):
    x, y, z = params
    return np.array([
        x + math.cos(y) - x ** 2 * math.sin(z ** 2) - 0.2,
        math.tan(x) - y + y * math.sin(z - 1) + 0.1,
        math.sin(x + y) + 2 * y + 2 * z - 0.1
    ])


# Система нелинейных уравнений приведенная к виду x = .., y = .., z = ..
def func_for_iter(params):
    x, y, z = params
    return np.array([
        0.2 - math.cos(y) + x * x * math.sin(z * z),
        math.tan(x) + y * math.sin(z - 1) + 0.1,
        (- math.sin(x + y) - 2 * y + 0.1) / 2
    ], float)


# Матрица Якоби
def matrix_jacobian(params):
    x, y, z = params
    return np.array([
        [1 - 2 * x * math.sin(z ** 2), -math.sin(y), -2 * z * (x ** 2) * math.cos(z ** 2)],
        [1 / (math.cos(x) ** 2), -1 + math.sin(z - 1), y * math.cos(z - 1)],
        [math.cos(x + y), 2 + math.cos(x + y), 2]
    ], float)


# Матрица Якоби для метода простых итераций
def matrix_jacobian_for_simple_iter(params):
    x, y, z = params
    return np.array([
        [2 * x * math.sin(z * z), math.sin(y), 2 * x * x * z * math.cos(z * z)],
        [(math.tan(x)) ** 2, math.sin(z - 1), y * math.cos(z - 1)],
        [(-math.cos(x + y)) / 2, (-math.cos(x + y)) / 2, 0]
    ], float)


# Метод Ньютона с точностью 1.0e-6
def method_Newton(dot):
    eps = 1.0e-6
    max_iter = 5
    f = func(dot)

    steps = 0
    while True:
        Jf = matrix_jacobian(dot)
        dot = dot - np.linalg.inv(Jf) @ f

        f = func(dot)
        if first_norm(f) < eps or steps >= max_iter:
            break
        steps += 1

    if steps > max_iter:
        print('Метод не сошелся, слишком много итераций для метода Ньютона.\n')
    return dot, steps


# Метод Простых итераций
def method_simple_iterations(dot):
    eps = 1.0e-3
    max_iter = 1000
    dot_prev = np.array(dot)

    for i in range(max_iter):
        dot = func_for_iter(dot_prev)

        if first_norm(dot - dot_prev) < eps:
            return dot, i
        dot_prev = dot

    print('Метод не сошелся, слишком много итераций.\n')


# Достаточное условие сходимости для метода простых итераций
def dost_usl_shodim_simple_iter(dot):
    max_sum = -np.inf

    for fi in matrix_jacobian_for_simple_iter(dot):
        max_sum = max(max_sum, np.sum(abs(fi)))

    if (max_sum < 1):
        print("Достаточное условие сходимости выполняется\n")
    else:
        print("\nДостаточное условие сходимости не выполняется\n")


# Вектор Невязки
def vector_nevyazki(f, dot):
    return np.array(f) - func(np.array(dot))


# Первая норма
def first_norm(x):
    return sum(abs(x.flatten()))


if __name__ == '__main__':
    # Точка Xo для начального приближения
    matrix_X0_dot_arr = [
        [-0.01],
        [0.03],
        [-0.6]
    ]
    matrix_X0_dot = np.array(matrix_X0_dot_arr, float)

    # 1) Решение системы нелинейных уравнений f(x) = 0 методом простых итераций с точностью 1e-3
    matrix_dot_solved, steps = method_simple_iterations(
        matrix_X0_dot.copy())

    print('Метод простых итераций')
    print_matrix(matrix_dot_solved)
    print('Шагов метода:', steps)

    # Вектор невязки
    print("\nВектор невязки")
    print_matrix(
        vector_nevyazki([[0], [0], [0]], matrix_dot_solved.copy()), "", 4,
        16)

    # 2) Проверка выполнения достаточного условия сходимости
    dost_usl_shodim_simple_iter([-0.015, 0.025, -0.65])

    # 3) Уточнение полученного решения методом Ньютона с точностью 1e-6
    matrix_dot_solved, steps = method_Newton(matrix_X0_dot.copy())

    print('Метод Ньютона')
    print_matrix(matrix_dot_solved)
    print('Шагов метода:', steps)

    # Вектор невязки
    print("\nВектор невязки")
    print_matrix(
        vector_nevyazki([[0], [0], [0]], matrix_dot_solved.copy()), "", 4,
        16)

    # Находим корни системы - Проверка с помощью sp.fsolve
    # matrix_dot_solved_arr = sp.fsolve(func, [1, 1, 1])
    # matrix_dot_solved = np.array(np.resize(matrix_dot_solved_arr, (3, 1)), float)
