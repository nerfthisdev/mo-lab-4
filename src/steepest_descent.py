from typing import Callable
import numpy as np


def golden_section_search(f, a, b, tol=1e-5, max_iter=1000):
    """Одномерная минимизация методом золотого сечения."""
    phi = (1 + np.sqrt(5)) / 2
    rho = phi - 1  # ~0.618
    x1 = b - rho * (b - a)
    x2 = a + rho * (b - a)
    f1, f2 = f(x1), f(x2)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - rho * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + rho * (b - a)
            f2 = f(x2)

    return (a + b) / 2


def steepest_descent(f, grad, x0, eps=1e-4, max_iter=1000, trace=False):
    x = np.array(x0, dtype=float)

    for k in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < eps:
            if trace:
                print(f"STOP: grad norm < eps at iter {k}")
            break

        d = -g / np.linalg.norm(g)  # направление

        phi = lambda alpha: f(x + alpha * d)
        alpha = golden_section_search(phi, 0, 1)  # ищем шаг в [0,1]

        x_new = x + alpha * d

        if trace:
            print(f"iter={k}, f(x)={f(x):.6f}, alpha={alpha:.4f}, x={x}")

        if np.linalg.norm(x_new - x) < eps:
            x = x_new
            break

        x = x_new

    return x, f(x)
