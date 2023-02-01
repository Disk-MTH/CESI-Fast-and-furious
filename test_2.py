import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

R = 0.115
g = 9.81
m = 0.03


def equation_diff(y, t, R, g, m, c):
    f, f_der = y
    f_double_der = -g * np.sin(f) - c * (-R * f_der ** 2 * m + g * np.cos(f) / m) / R
    return [f_der, f_double_der]


y0 = [0, 0]
t = np.arange(0, 1, 0.01)
c = 0.1
sol = odeint(equation_diff, y0, t, args=(R, g, m, c))
f = sol[:, 0]
f_der = sol[:, 1]

plt.plot(t, f)
plt.plot(t, f_der)
plt.show()
