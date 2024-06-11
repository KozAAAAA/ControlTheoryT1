import numpy as np
from .controller import Controller
from models.manipulator_model import ManipulatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.models = [
            ManipulatorModel(Tp, 0.1, 0.05),
            ManipulatorModel(Tp, 0.01, 0.01),
            ManipulatorModel(Tp, 1.0, 0.3),
        ]
        self.i = 0
        self.u = np.zeros((2, 1))
        self.x = np.zeros((4, 1))
        self.Tp = Tp
        self.KP = np.array([[1, 0], [0, 1]])
        self.KD = np.array([[1, 0], [0, 1]])

    def choose_model(self, x):
        err = []
        for model in self.models:
            M = model.M(x)
            C = model.C(x)

            q_dot = x[2:]
            q_ddot_pred = np.linalg.inv(M) @ (self.u - C @ q_dot[:, np.newaxis])
            q_ddot_actual = (q_dot - self.x[2:]) / self.Tp

            err.append(np.linalg.norm(q_ddot_actual - q_ddot_pred.squeeze()))

        self.i = np.argmin(err)
        print(f"Model {self.i + 1} chosen")

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        v = q_r_ddot - self.KP @ (q - q_r) - self.KD @ (q_dot - q_r_dot)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        self.x = x
        self.u = u
        return u
