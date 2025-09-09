from typing import Callable
import numpy as np
from function import grad


def gradient_descent(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    h: float = 0.25,
    max_iter: int = 5000,
    eps: float = 1e-4,
    trace: bool = False,
):
    x = np.array(x0, dtype=float)  # копия начальной точки
    it = 0
    f_old = f(x)

    print()
    print("!!! GRADIENT DESCENT !!!")

    while it < max_iter:
        grad_value = grad(x)

        # шаг градиентного спуска
        x[0] = x[0] - h * grad_value[0]
        x[1] = x[1] - h * grad_value[1]

        f_new = f(x)

        if trace:
            print(
                f"iter={it:4d}, f(x)={f_new:.6f}, x={x}, grad={grad_value}, h={h:.4f}"
            )

        # критерий остановки
        if abs(f_new - f_old) < eps:
            if trace:
                print(f"STOP: |f_new - f_old| < eps at iter {it}")
            break
        if f_new >= f_old:
            h *= 0.8
            if trace:
                print(f"  f_new >= f_old, уменьшаем шаг h -> {h:.4f}")

        f_old = f_new
        it += 1

    return x, f(x), it
