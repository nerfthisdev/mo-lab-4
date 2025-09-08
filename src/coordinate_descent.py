from typing import Callable
import numpy as np


def coordinate_descent(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    max_iter: int = 5000,
    eps: float = 1e-4,
    trace: bool = False,
):
    x = np.array(x0, dtype=float)  # копия начальной точки
    it = 0

    if trace:
        print(f"M0 = ({x[0]:.6f}, {x[1]:.6f}), f(M0) = {f(x):.6f}")

    while it < max_iter:
        x_old = x.copy()

        if it % 2 == 0:
            # чётная итерация → обновляем x1 при фиксированном x2
            x[0] = (1 - 3 * x[1]) / 2.0
            fixed = "x2"
            updated = "x1"
        else:
            # нечётная итерация → обновляем x2 при фиксированном x1
            x[1] = (2 - 3 * x[0]) / 6.0
            fixed = "x1"
            updated = "x2"

        it += 1

        if trace:
            print(
                f"M{it} = ({x[0]:.6f}, {x[1]:.6f}), "
                f"f(M{it}) = {f(x):.6f} "
                f"(фиксирую {fixed}, пересчитываю {updated})"
            )

        # критерий останова: проверяем, насколько изменилась точка
        delta = ((x[0] - x_old[0]) ** 2 + (x[1] - x_old[1]) ** 2) ** 0.5
        if delta < eps:
            print("!!! delta < eps !!!")
            break

    return x, f(x), it
