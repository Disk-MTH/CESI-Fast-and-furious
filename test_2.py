import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

radius = 0.115
g = 9.81
mu = 0.03
theta0 = [np.pi / 2, 37]

def equation_diff(y, t):
    f, fp = y
    fpp = - (g * np.sin(f) / radius) - (mu * fp ** 2) - (g * np.cos(f) * mu / radius)

    return [fp, fpp]


t = np.arange(0, 2, 0.01)
sol = odeint(equation_diff, theta0, t)
theta = sol[:, 0]
theta_p = sol[:, 1]

plt.plot(t, theta, label="theta : angular position")
plt.plot(t, theta_p, label="theta_p : angular velocity")
plt.legend()
plt.show()
