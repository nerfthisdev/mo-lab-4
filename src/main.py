import numpy as np

from coordinate_descent import coordinate_descent
from function import f, grad
from gradient_descent import gradient_descent
from steepest_descent import steepest_descent

x0 = np.array([3.0, 3.0])

# print(f(np.array([-4.0, 3.0])))

print(coordinate_descent(f, x0, trace=True))
print(gradient_descent(f, x0, h=0.25))
print(steepest_descent(f, grad, x0, trace=True))
