import numpy as np
import math


class ManipulatorModelF:
    def __init__(self, Tp):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.04
        self.m1 = 3.0
        self.l2 = 0.4
        self.r2 = 0.04
        self.m2 = 2.4
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1**2 + self.l1**2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2**2 + self.l2**2)
        self.m3 = 1.0
        self.r3 = 0.05
        self.I_3 = 2.0 / 5 * self.m3 * self.r3**2

    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        _, q2, _, _ = x
        d1 = self.l1 / 2
        d2 = self.l2 / 2
        cos2 = np.cos(q2)
        a = (
            self.m1 * d1**2
            + self.I_1
            + self.m2 * (self.l1**2 + d2**2)
            + self.I_2
            + self.m3 * (self.l1**2 + self.l2**2)
            + self.I_3
        )
        b = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        g = self.m2 * d2**2 + self.I_2 + self.m3 * self.l2**2 + self.I_3

        M = [[a + 2 * b * cos2, g + b * cos2], [g + b * cos2, g]]
        return M

    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        _, q2, q1_dot, q2_dot = x

        d2 = self.l2 / 2

        sin2 = math.sin(q2)

        b = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2

        C = [
            [-b * sin2 * q2_dot, -b * sin2 * (q1_dot + q2_dot)],
            [b * sin2 * q1_dot, 0],
        ]
        return C


class ManipulatorModel:
    def __init__(self, Tp, m3, r3):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.01
        self.m1 = 1.0
        self.l2 = 0.5
        self.r2 = 0.01
        self.m2 = 1.0
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1**2 + self.l1**2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2**2 + self.l2**2)
        self.m3 = m3
        self.r3 = r3
        self.I_3 = 2.0 / 5 * self.m3 * self.r3**2

    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        _, q2, _, _ = x
        d1 = self.l1 / 2
        d2 = self.l2 / 2
        cos2 = np.cos(q2)
        a = (
            self.m1 * d1**2
            + self.I_1
            + self.m2 * (self.l1**2 + d2**2)
            + self.I_2
            + self.m3 * (self.l1**2 + self.l2**2)
            + self.I_3
        )
        b = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        g = self.m2 * d2**2 + self.I_2 + self.m3 * self.l2**2 + self.I_3

        M = [[a + 2 * b * cos2, g + b * cos2], [g + b * cos2, g]]
        return M

    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        _, q2, q1_dot, q2_dot = x

        d2 = self.l2 / 2

        sin2 = math.sin(q2)

        b = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2

        C = [
            [-b * sin2 * q2_dot, -b * sin2 * (q1_dot + q2_dot)],
            [b * sin2 * q1_dot, 0],
        ]
        return C


class ManipulatorModel:
    def __init__(self, Tp, m3, r3):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.01
        self.m1 = 1.0
        self.l2 = 0.5
        self.r2 = 0.01
        self.m2 = 1.0
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1**2 + self.l1**2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2**2 + self.l2**2)
        self.m3 = m3
        self.r3 = r3
        self.I_3 = 2.0 / 5 * self.m3 * self.r3**2

    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        _, q2, _, _ = x
        d1 = self.l1 / 2
        d2 = self.l2 / 2
        cos2 = np.cos(q2)
        a = (
            self.m1 * d1**2
            + self.I_1
            + self.m2 * (self.l1**2 + d2**2)
            + self.I_2
            + self.m3 * (self.l1**2 + self.l2**2)
            + self.I_3
        )
        b = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        g = self.m2 * d2**2 + self.I_2 + self.m3 * self.l2**2 + self.I_3

        M = [[a + 2 * b * cos2, g + b * cos2], [g + b * cos2, g]]
        return M

    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        _, q2, q1_dot, q2_dot = x

        d2 = self.l2 / 2

        sin2 = math.sin(q2)

        b = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2

        C = [
            [-b * sin2 * q2_dot, -b * sin2 * (q1_dot + q2_dot)],
            [b * sin2 * q1_dot, 0],
        ]
        return C
