import numpy as np


def f(x: np.ndarray) -> float:
    z = x[0] ** 2 + (3 * x[1] ** 2) + 3 * x[0] * x[1] - x[0] - 2 * x[1] - 1
    return z


def grad(x: np.ndarray) -> np.ndarray:
    df_dx1 = 2 * x[0] + 3 * x[1] - 1
    df_dx2 = 6 * x[1] + 3 * x[0] - 2
    return np.array([df_dx1, df_dx2], dtype=float)
