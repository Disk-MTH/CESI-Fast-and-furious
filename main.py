import matplotlib.pyplot as plt
import numpy as np


class Slope:
    g = 9.81

    def __init__(self, duration, interval, alpha, v0, p0, mu=0.0, is_alpha_degrees=False):

        if duration <= 0.0:
            raise ValueError("duration must be positive")

        if interval <= 0.0:
            raise ValueError("interval must be positive")

        if mu < 0.0 or mu >= 1.0:
            raise ValueError("mu must be in [0.0 ; 1[")

        if is_alpha_degrees:
            alpha = np.radians(alpha)

        self.time = np.arange(0, duration, interval)
        self.alpha = alpha
        self.v0 = v0
        self.p0 = p0
        self.mu = mu

    def a_at_t(self):
        if self.mu == 0.0:
            ax = self.g * np.sin(self.alpha)
        else:
            ax = self.g * np.sin(self.alpha) * (1 - self.mu)

        ay = 0

        return ax, ay, np.sqrt(ax) ** 2 + ay ** 2

    def a(self):
        return [self.a_at_t()[2] for t in self.time]

    def v_at_t(self, t):
        if self.mu == 0.0:
            vx = self.g * np.sin(self.alpha) * t + self.v0[0]
        else:
            vx = self.g * np.sin(self.alpha) * (1 - self.mu) * t + self.v0[0]

        vy = self.v0[1]

        return vx, vy, np.sqrt(vx ** 2 + vy ** 2)

    def v(self):
        return [self.v_at_t(t)[2] for t in self.time]

    def p_at_t(self, t):
        if self.mu == 0.0:
            px = self.g * np.sin(self.alpha) * t**2 / 2 + self.v0[0] * t + self.p0[0]
        else:
            px = self.g * np.sin(self.alpha) * (1 - self.mu) * t**2 / 2 + self.v0[0] * t + self.p0[0]

        py = self.v0[1] * t + self.p0[1]

        return px, py, np.sqrt(px ** 2 + py ** 2)

    def p(self):
        return [self.p_at_t(t)[2] for t in self.time]

    def trace(self):
        plt.plot(self.time, self.a(), label="Acceleration")
        plt.plot(self.time, self.v(), label="Velocity")
        plt.plot(self.time, self.p(), label="Position")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    slope = Slope(10, 0.1, 40, (0, 0), (0, 0), mu=0.002, is_alpha_degrees=True)
    slope.trace()
