from models.manipulator_model import ManipulatorModel
from .controller import Controller
import numpy as np

KP = 1
KD = 1

class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManipulatorModel(Tp, 1.0, 0.05)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """

        (q1, q2, q1_dot, q2_dot) = x
        v = q_r_ddot + KP * (q_r - [q1, q2]) + KD * (q_r_dot - [q1_dot, q2_dot])
        u = self.model.M(x) @ v + self.model.C(x) @ np.array([q1_dot, q2_dot])
        return u
