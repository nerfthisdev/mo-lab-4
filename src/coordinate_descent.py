from typing import Callable
import numpy as np


def coordinate_descent(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    max_iter: int = 500,
    eps: float = 0.01,
):
    return 0
