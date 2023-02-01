import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
import random


class MechanicalStudy:
    g = 9.81

    def __init__(self, duration, interval, v0, p0):
        if duration <= 0.0:
            raise ValueError("duration must be positive")

        if interval <= 0.0:
            raise ValueError("interval must be positive")

        self.time = np.arange(0, duration + interval, interval)
        self.v0 = v0
        self.p0 = p0


class Slope(MechanicalStudy):
    def __init__(self, duration, interval, v0, p0, alpha, mu=0.0, is_alpha_degrees=False):
        """
        Slope simulation class constructor
        :param duration: duration of the simulation in seconds
        :param interval:  interval between two points in seconds
        :param alpha: angle of the slope in radians (or degrees if is_alpha_degrees is True)
        :param v0: tuple of the initial velocity (vx, vy)
        :param p0: tuple of the initial position (px, py)
        :param mu: coefficient of friction (0.0 for no friction)
        :param is_alpha_degrees: True if alpha is in degrees, False if alpha is in radians
        """

        super().__init__(duration, interval, v0, p0)

        if is_alpha_degrees:
            alpha = np.radians(alpha)

        if mu < 0.0 or mu >= 1.0:
            raise ValueError("mu must be in [0.0 ; 1[")

        self.alpha = alpha
        self.mu = mu

    def a(self):
        """
        Return the acceleration at time t (in m/s²)
        :return: tuple of the acceleration (ax, ay, a)
        """

        ax = self.g * np.sin(self.alpha) * (1 - self.mu)
        ay = 0

        return ax, ay, np.sqrt(ax ** 2 + ay ** 2)

    def v(self, t):
        """
        Return the velocity at time t (in m/s)
        :param t: time in seconds
        :return: tuple of the velocity (vx, vy, v)
        """

        if self.mu == 0.0:
            vx = self.g * np.sin(self.alpha) * t + self.v0[0]
        else:
            vx = self.g * np.sin(self.alpha) * (1 - self.mu) * t + self.v0[0]

        vy = self.v0[1]

        return vx, vy, np.sqrt(vx ** 2 + vy ** 2)

    def p(self, t):
        """
        Return the position at time t (in m)
        :param t: time in seconds
        :return: tuple of the position (px, py, p)
        """

        if self.mu == 0.0:
            px = self.g * np.sin(self.alpha) * t**2 / 2 + self.v0[0] * t + self.p0[0]
        else:
            px = self.g * np.sin(self.alpha) * (1 - self.mu) * t**2 / 2 + self.v0[0] * t + self.p0[0]

        py = self.v0[1] * t + self.p0[1]

        return px, py, np.sqrt(px ** 2 + py ** 2)

    def get_values_on_interval(self):
        """
        Return the values of the acceleration, velocity and position on the interval
        :return: tuple of the acceleration, velocity and position
        """

        return [self.a()[2] for t in self.time], \
            [self.v(t)[2] for t in self.time], \
            [self.p(t)[2] for t in self.time]

    def get_t_at_p(self, p):
        """
        Return the time at which the position is p (in s)
        :param p:  position in meters
        :return: time in seconds
        """

        if self.mu == 0.0:
            return np.sqrt(2 * p / (self.g * np.sin(self.alpha)))
        else:
            return np.sqrt(2 * p / (self.g * np.sin(self.alpha) * (1 - self.mu)))

    def get_v_end_terms_of_height(self, height):
        """
        Return the velocity at the given height (in m/s)
        :param height: height in meters
        :return: velocity as a tuple (vx, vy, v)
        """

        if height < 0:
            raise ValueError("height must be positive")

        return self.v(self.get_t_at_p(height / np.sin(self.alpha)))

    def annotate_point(self, t, color):
        """
        Annotate the point at time i with the acceleration, velocity and position
        :param t: time in seconds
        :param color: color of the annotation
        """

        closest_index = (np.abs(self.time - t)).argmin()
        closest = [
            (self.get_values_on_interval()[0][closest_index], 'a'),
            (self.get_values_on_interval()[1][closest_index], 'v'),
            (self.get_values_on_interval()[2][closest_index], 'p'),
        ]

        for closest_val, closest_label in closest:
            plt.scatter(t, closest_val, color=color)
            plt.annotate(f'{closest_label} = {closest_val:.2f}', (t, closest_val), textcoords='offset points',
                         xytext=(10, -10), ha='center', color=color)
            plt.annotate(f't = {t:.2f}', (t, closest_val), textcoords='offset points',
                         xytext=(10, -20), ha='center', color=color)

    def trace(self, *args):
        """
        Trace the acceleration, velocity and position
        :param args: time(s) in seconds to annotate
        """

        plt.plot(self.time, self.get_values_on_interval()[0], label="Acceleration (m/s^2)")
        plt.plot(self.time, self.get_values_on_interval()[1], label="Velocity (m/s)")
        plt.plot(self.time, self.get_values_on_interval()[2], label="Position (m)")

        for i in args:
            color = "#" + ''.join([random.choice('0123456789ABCDEF') for i in range(6)])

            if i > self.time[-1] or i < 0:
                raise ValueError("t must be in [0 ; {}]".format(self.time[-1]))

            plt.axvline(x=i, color=color)
            self.annotate_point(i, color)

        plt.legend()
        plt.show()

    def trace_v_end_terms_of_height(self, ceil, interval):
        """
        Trace the velocity as a function of the height
        :param ceil: maximum height in meters
        :param interval: interval between each height in meters
        """

        if interval <= 0:
            raise ValueError("interval must be positive")

        height = np.arange(0, ceil + interval, interval)
        plt.plot(height, [self.get_v_end_terms_of_height(h)[2] for h in height], label="Velocity (m/s)")
        plt.legend()
        plt.show()


class Looping(MechanicalStudy):
    def __init__(self, duration, interval, radius, mass, theta_0, v0, p0, mu=0.0, is_theta_degrees=False):
        """
        Looping simulation class constructor
        :param duration: duration of the simulation in seconds
        :param interval: interval between two points in seconds
        :param radius: radius of the loop in meters
        :param mass: mass of the car in kilograms
        :param theta_0: tuple of initial angle and initial angular velocity
        :param v0: tuple of the initial velocity (vx, vy)
        :param p0: tuple of the initial position (px, py)
        :param mu: coefficient of friction (0.0 for no friction)
        :param is_theta_degrees: True if theta_0 is in degrees, False if theta_0 is in radians
        """

        super().__init__(duration, interval, v0, p0)

        if radius <= 0.0:
            raise ValueError("radius must be positive")

        if mass <= 0.0:
            raise ValueError("mass must be positive")

        if is_theta_degrees:
            theta_0 = np.radians(theta_0[0]), np.radians(theta_0[1])

        if mu < 0.0 or mu >= 1.0:
            raise ValueError("mu must be in [0.0 ; 1[")

        self.mass = mass
        self.radius = radius
        self.theta_0 = theta_0
        self.mu = mu

    def build_equation(self, y, t):
        f, f_p = y

        return [
            f_p,
            -self.g * np.sin(f) - self.mu * (-self.radius * f_p ** 2 * self.mass + self.g * np.cos(f) / self.mass) / self.radius
        ]

    def solv_equation(self):
        return odeint(self.build_equation, self.theta_0, self.time)[:, 0]

    def trace(self):
        plt.plot(self.time, self.solv_equation())
        plt.show()


if __name__ == "__main__":
    slope = Slope(1, 0.01, 40, (0, 0), (0, 0), mu=0.002, is_alpha_degrees=True)

    #duration, interval, radius, mass, theta_0, v0, p0, mu=0.0, is_theta_degrees=False

    looping = Looping(5, 0.01, 0.115, 0.03, (0, 0), (0, 0), (0, 0), mu=0.002, is_theta_degrees=True)

    looping.trace()


