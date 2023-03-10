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
    def __init__(self, duration, interval, alpha, v0, p0, mu=0.0, is_alpha_degrees=False):
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

        ax = self.g * (np.sin(self.alpha) - self.mu * np.cos(self.alpha))
        ay = 0

        return ax, ay, np.sqrt(ax ** 2 + ay ** 2)

    def v(self, t):
        """
        Return the velocity at time t (in m/s)
        :param t: time in seconds
        :return: tuple of the velocity (vx, vy, v)
        """

        vx = (self.g * (np.sin(self.alpha) - self.mu * np.cos(self.alpha))) * t + self.v0[0]
        vy = self.v0[1]

        return vx, vy, np.sqrt(vx ** 2 + vy ** 2)

    def p(self, t):
        """
        Return the position at time t (in m)
        :param t: time in seconds
        :return: tuple of the position (px, py, p)
        """

        px = (self.g * (np.sin(self.alpha) - self.mu * np.cos(self.alpha))) * t ** 2 / 2 + self.v0[0] * t + self.p0[0]
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

        a = self.g * (np.sin(self.alpha) - self.mu * np.cos(self.alpha)) / 2
        b = self.v0[0]
        c = self.p0[0] - p

        delta = b ** 2 - 4 * a * c

        if delta < 0:
            raise ValueError("no solution")

        return (-b + np.sqrt(delta)) / (2 * a)

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
        """"
        Trace the path of the object
        :param args: Times to annotate
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

        plt.title("Acceleration, velocity and position as a function of time")
        plt.legend()
        plt.show()

    def trace_v_end_terms_of_height(self, limits, interval):
        """
        Trace the velocity as a function of the height
        :param limits: tuple of minimum and maximum height in meters
        :param interval: interval between each height in meters
        """

        if interval <= 0:
            raise ValueError("interval must be positive")

        height = np.arange(limits[0], limits[1] + interval, interval)
        plt.plot(height, [self.get_v_end_terms_of_height(h)[2] for h in height], label="Velocity with friction (m/s)")
        self.mu = 0.0
        plt.plot(height, [self.get_v_end_terms_of_height(h)[2] for h in height], label="Velocity without friction (m/s)")
        plt.title("Velocity as a function of the height")
        plt.gca().set_xlabel("Height (m)")
        plt.gca().set_ylabel("Velocity (m/s)")
        plt.legend()
        plt.show()


class Looping(MechanicalStudy):
    def __init__(self, duration, interval, radius, v0, p0, mu=0.0, is_v0_ms=False, is_p0_degrees=False):
        """
        Looping simulation class constructor
        :param duration: duration of the simulation in seconds
        :param interval: interval between two points in seconds
        :param radius: radius of the loop in meters
        :param v0: initial velocity in rad/s
        :param p0: initial angle in radians
        :param mu: coefficient of friction (0.0 for no friction)
        :param is_v0_ms: True if v0 is in m/s, False if v0 is in rad/s
        :param is_p0_degrees: True if p0 is in degrees, False if p0 is in radians
        """

        super().__init__(duration, interval, v0, p0)

        if radius <= 0.0:
            raise ValueError("radius must be positive")

        if is_v0_ms:
            self.v0 = self.v0 / radius

        if is_p0_degrees:
            self.p0 = np.radians(self.p0)

        if mu < 0.0 or mu >= 1.0:
            raise ValueError("mu must be in [0.0 ; 1[")

        self.radius = radius
        self.mu = mu

    def build_equation(self, y, t):
        """
        Build the differential equation
        :param y: The current state of the system
        :param t: The timeline
        :return: The system to solve
        """

        f, fp = y

        return [
            fp,
            - (self.g * np.sin(f) / self.radius) - (self.mu * fp ** 2) - (
                    self.g * np.cos(f) * self.mu / self.radius)
        ]

    def solv_equation(self):
        """
        Solve the differential equation
        :return: The solution of the differential equation
        """

        return odeint(self.build_equation, (self.p0, self.v0), self.time)

    def get_min_velocity(self, precision=0.001, v_in_ms=False):
        """
        Get the minimum velocity to pass looping with dichotomy
        :param precision: precision of the result
        :param v_in_ms: True if the result must be in m/s, False if the result must be in rad/s
        :return: the minimum velocity to pass the looping
        """

        original = (self.v0, self.p0)
        low = 0
        high = self.v0
        while (high - low) > precision:
            mid = (low + high) / 2
            self.v0 = mid
            result = self.solv_equation()
            if np.sin(result[-1][0]) < 0:
                high = mid
            else:
                low = mid

        self.v0, self.p0 = original
        return low if not v_in_ms else low * self.radius

    def get_v_after_looping(self, v_in_ms=False):
        """
        Get the velocity after the first looping
        :param v_in_ms: True if the result must be in m/s, False if the result must be in rad/s
        :return: the velocity after the first looping
        """
        for i in range(len(self.solv_equation()[:, 0])):
            if self.solv_equation()[:, 0][i] > self.p0 + 2 * np.pi:
                if not v_in_ms:
                    return self.solv_equation()[:, 1][i]
                else:
                    return self.solv_equation()[:, 1][i] * self.radius

    def trace(self, p_in_degrees=False, v_in_ms=False):
        """
        Trace the angular position and angular velocity
        :param p_in_degrees: True if the angular position must be in
            degrees, False if the angular position must be in radians
        :param v_in_ms: True if the angular velocity must
            be in m/s, False if the angular velocity must be in rad/s
        """

        fig, (row_1, row_2) = plt.subplots(2, 1)
        fig.subplots_adjust(hspace=0.5)

        if p_in_degrees:
            row_1.plot(self.time, np.degrees(self.solv_equation()[:, 0]), label="Angular position (deg)")
        else:
            row_1.plot(self.time, self.solv_equation()[:, 0], label="Angular position (rad)")
        row_1.set_title("Angular position as a function of time")
        row_1.legend()

        if v_in_ms:
            row_2.plot(self.time, self.solv_equation()[:, 1] * self.radius, label="Angular velocity (m/s)")
        else:
            row_2.plot(self.time, self.solv_equation()[:, 1], label="Angular velocity (rad/s)")
        row_2.set_title("Angular velocity as a function of time")
        row_2.legend()

        plt.show()


class Ravine(MechanicalStudy):
    def __init__(self, duration, interval, ravine_dim, mass, v0, p0, sc_x=0.0, sc_z=0.0, rho=1.225):
        """
        Ravine simulation class constructor
        :param duration: duration of the simulation in seconds
        :param interval: interval between two points in seconds
        :param ravine_dim: dimensions of the ravine in meters (width, height)
        :param mass: mass of the object in kilograms
        :param v0: initial velocity in m/s
        :param p0: initial position in meters
        :param sc_x: coefficient of friction for drag (surface * coefficient of drag)
        :param sc_z: coefficient of friction for lift (surface * coefficient of lift)
        :param rho: density of the air in kg/m^3
        """

        super().__init__(duration, interval, v0, p0)

        if ravine_dim[0] <= 0.0 or ravine_dim[1] <= 0.0:
            raise ValueError("Ravine dimensions must be positive")

        if sc_x < 0.0 or sc_z < 0.0:
            raise ValueError("Coefficient of friction must be positive or null")

        self.ravine_dim = ravine_dim
        self.mass = mass
        self.sc_x = sc_x
        self.sc_z = sc_z
        self.rho = rho

    def a_no_friction(self):
        """
        Get the acceleration of the object
        :return: the acceleration of the object in m/s^2
        """

        if self.sc_z > 0.0 or self.sc_x > 0.0:
            raise ValueError("Coefficient of friction must be null for no friction simulation")

        ax = 0
        ay = - self.g

        return ax, ay, np.sqrt(ax ** 2 + ay ** 2)

    def v_no_friction(self, t):
        """
        Get the velocity of the object
        :param t: time in seconds
        :return: the velocity of the object in m/s
        """

        if self.sc_z > 0.0 or self.sc_x > 0.0:
            raise ValueError("Coefficient of friction must be null for no friction simulation")

        vx = self.v0[0]
        vy = - self.g * t + self.v0[1]

        return vx, vy, np.sqrt(vx ** 2 + vy ** 2)

    def p_no_friction(self, t):
        """
        Get the position of the object
        :param t: time in seconds
        :return: the position of the object in m
        """

        if self.sc_z > 0.0 or self.sc_x > 0.0:
            raise ValueError("Coefficient of friction must be null for no friction simulation")

        px = self.v0[0] * t + self.p0[0]
        py = - self.g / 2 * t ** 2 + self.v0[1] * t + self.p0[1]

        return px, py, np.sqrt(px ** 2 + py ** 2)

    def get_values_on_interval(self):
        """
        Return the values of the acceleration, velocity and position on the interval
        :return: tuple of the acceleration, velocity and position
        """

        return [self.a_no_friction()[2] for t in self.time], \
            [self.v_no_friction(t)[2] for t in self.time], \
            [self.p_no_friction(t)[2] for t in self.time]

    def build_equation(self, y, t):
        """
        Build the differential equation
        :param y: The current state of the system
        :param t: The timeline
        :return: The system to solve
        """

        x, y, vx, vy = y
        v = np.sqrt(vx ** 2 + vy ** 2)

        return [
            vx, vy,
            (- self.rho / 2 * self.sc_x * vx * v - self.rho / 2 * self.sc_z * vy * v) / self.mass,
            (- self.rho / 2 * self.sc_x * vy * v + self.rho / 2 * self.sc_z * vx * v) / self.mass - self.g
        ]

    def solv_equation(self):
        """
        Solve the differential equation
        :return: The solution of the differential equation
        """

        return odeint(self.build_equation, (self.p0[0], self.p0[1], self.v0[0], self.v0[1]), self.time)


    def trace(self):
        """
        Trace the path of the object
        """

        x, y = self.solv_equation()[:, 0], self.solv_equation()[:, 1]

        x_with_y = (np.abs(np.array(y, dtype=float) + self.ravine_dim[1])).argmin()
        x_1 = x[x_with_y]
        y_1 = y[x_with_y]
        plt.axvline(x=x_1, color="red", label="Impact depending on max Y")
        plt.axhline(y=y_1, color="orange", label="Max Y")
        plt.scatter(x_1, y_1, color="black")
        plt.annotate(f'X = {x_1:.2f}', (x_1 - 0.15, y_1 - 0.015), textcoords='offset points',
                     xytext=(10, -10), ha='center', color="red")
        plt.annotate(f'Y = {y_1:.2f}', (x_1 - 0.15, y_1 - 0.015), textcoords='offset points',
                     xytext=(10, -20), ha='center', color="orange")

        print(f"X = {x_1} , Y = {y_1}")

        y_for_x = (np.abs(np.array(x, dtype=float) - self.ravine_dim[0])).argmin()
        x_2 = x[y_for_x]
        y_2 = y[y_for_x]
        plt.axvline(x=x[y_for_x], color="green", label="Max X")
        plt.axhline(y=y[y_for_x], color="blue", label="Impact depending on max X")
        plt.scatter(x_2, y_2, color="black")
        plt.annotate(f'X = {x_2:.2f}', (x_2 - 0.15, y_2 - 0.015), textcoords='offset points',
                     xytext=(10, -10), ha='center', color="green")
        plt.annotate(f'Y = {y_2:.2f}', (x_2 - 0.15, y_2 - 0.015), textcoords='offset points',
                     xytext=(10, -20), ha='center', color="blue")

        print(f"X = {x_2} , Y = {y_2}")

        plt.plot(x, y, label="Path with friction (m)")

        x, y = [], []

        sc_x = self.sc_x
        sc_z = self.sc_z

        self.sc_x = 0.0
        self.sc_z = 0.0

        for t in self.time:
            x.append(self.p_no_friction(t)[0])
            y.append(self.p_no_friction(t)[1])

        plt.plot(x, y, label="Path without friction (m)")

        plt.gca().set_xlabel("X position (m)")
        plt.gca().set_ylabel("Y position (m)")
        plt.legend()

        self.sc_x = sc_x
        self.sc_z = sc_z

        plt.show()


if __name__ == "__main__":
    """Generic parameters"""
    study_duration = 1  # in seconds
    study_interval = 0.01  # in seconds
    car_mass = 0.03  # in kg
    friction_coefficient = 0.002  # coefficient of friction with the ground

    """Slope parameters"""
    slope_height = 0.93  # in meters
    slope_angle = 40  # in degrees
    slope_v0 = (0, 0)  # v0x, v0y in m/s
    slope_p0 = (0, 0)  # p0x, p0y in m

    slope_limits = (0, 5)  # in meters
    slope_height_interval = 0.01  # in meters

    """Looping parameters"""
    looping_radius = 0.115  # in meters
    looping_p0 = 0  # in degrees

    """Ravine parameters"""
    ravine_length = 0.7  # in meters
    ravine_height = 0.1  # in meters
    ravine_drag_coefficient = 0.001  # coefficient of drag * surface
    ravine_lift_coefficient = 0.001  # coefficient of lift * surface

    # Create the slope object
    slope = Slope(duration=study_duration, interval=study_interval, alpha=slope_angle, v0=slope_v0, p0=slope_p0,
                  mu=friction_coefficient, is_alpha_degrees=True)

    slope.trace(slope.get_t_at_p(slope_height / np.sin(np.radians(slope_angle))))
    slope.trace_v_end_terms_of_height(limits=slope_limits, interval=slope_height_interval)

    v_at_slope_end = slope.get_v_end_terms_of_height(slope_height)[2]
    print("Velocity at the end of the slope: {} m/s".format(v_at_slope_end))

    # Create the looping object
    looping = Looping(duration=study_duration, interval=study_interval, radius=looping_radius, v0=v_at_slope_end,
                      p0=looping_p0, mu=friction_coefficient, is_v0_ms=True, is_p0_degrees=True)

    looping.trace()
    looping.trace(p_in_degrees=True, v_in_ms=True)
    print("Minimum velocity for looping: {} m/s".format(looping.get_min_velocity(v_in_ms=True)))
    print("Velocity at the end of the looping: {} m/s".format(looping.get_v_after_looping(v_in_ms=True)))

    # Find the height of the slope to have just the right velocity to pass the looping
    while looping.get_min_velocity(v_in_ms=True) < v_at_slope_end:
        slope_height -= 0.1
        v_at_slope_end = slope.get_v_end_terms_of_height(slope_height)[2]

    print("Minimum height of the slope to pass looping: {} m".format(slope_height))

    # Create the ravine object
    ravine = Ravine(duration=study_duration, interval=study_interval, ravine_dim=(ravine_length, ravine_height),
                    mass=car_mass, v0=(looping.get_v_after_looping(v_in_ms=True), 0), p0=(0, 0),
                    sc_x=ravine_drag_coefficient, sc_z=ravine_lift_coefficient)
    ravine.trace()
