from typing import Tuple, Any
import numpy as np

from coordinate_descent import coordinate_descent
from function import f, grad
from gradient_descent import gradient_descent
from steepest_descent import steepest_descent


# ----- утилиты форматирования -----


def fmt_vec(x: np.ndarray) -> str:
    # вектор с фиксированным числом знаков после запятой
    return f"({x[0]: .6f}, {x[1]: .6f})"


def fmt_num(x: float, width: int = 12, prec: int = 6) -> str:
    """
    Печатает число в фиксированном формате (без e-нотации),
    аккуратно обрезая хвостовые нули и точку. Выравнивает по правому краю.
    """
    s = f"{x:.{prec}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.rjust(width)


# ----- унификация результатов -----


def unpack(result: Any) -> Tuple[np.ndarray, float, int | None]:
    """
    Унифицирует кортеж результата:
    - (x, f) -> (x, f, None)
    - (x, f, iters) -> (x, f, iters)
    """
    if isinstance(result, tuple):
        if len(result) == 3:
            x, fx, iters = result
            return np.asarray(x, dtype=float), float(fx), int(iters)
        elif len(result) == 2:
            x, fx = result
            return np.asarray(x, dtype=float), float(fx), None
    # на всякий случай — если вернули только x
    x = np.asarray(result, dtype=float)
    return x, float(f(x)), None


# ----- запуск и сводка -----


def run_all(x0: np.ndarray, trace_methods: bool = False):
    # точное решение (для сравнения): ∇f(x*) = 0
    x_star = np.array([0.0, 1.0 / 3.0])
    f_star = -4.0 / 3.0

    # опционально подавим экспоненциальную нотацию у numpy-массивов
    np.set_printoptions(suppress=True)

    # запуски методов
    cd_res = unpack(coordinate_descent(f, x0, trace=trace_methods))
    gd_res = unpack(gradient_descent(f, x0, h=0.25, trace=trace_methods))
    sd_res = unpack(steepest_descent(f, grad, x0, trace=trace_methods))

    rows = []
    for name, (xk, fk, iters) in [
        ("Coordinate descent", cd_res),
        ("Gradient descent", gd_res),
        ("Steepest descent", sd_res),
    ]:
        gk_norm = float(np.linalg.norm(grad(xk)))
        dx = float(np.linalg.norm(xk - x_star))
        df = abs(fk - f_star)
        rows.append((name, xk, fk, gk_norm, iters, dx, df))

    # печать сводки
    header = (
        f"{'Method':<20} {'x':<40} {'f(x)':>12} {'||grad||':>12} "
        f"{'iters':>7} {'|x-x*|':>12} {'|f-f*|':>12}"
    )
    print("\n===== SUMMARY =====")
    print(header)
    print("-" * len(header))
    for name, xk, fk, gk_norm, iters, dx, df in rows:
        it_str = f"{iters}" if iters is not None else "-"
        print(
            f"{name:<20} {fmt_vec(xk):<40} "
            f"{fmt_num(fk, 12, 6)} {fmt_num(gk_norm, 12, 6)} {it_str:>7} "
            f"{fmt_num(dx, 12, 6)} {fmt_num(df, 12, 6)}"
        )

    print("\nExact solution:")
    print(f"  x* = {fmt_vec(x_star)},  f* = {f_star:.6f}")


if __name__ == "__main__":
    x0 = np.array([3.0, 3.0])
    # подробные логи методов: trace_methods=True
    run_all(x0, trace_methods=True)

