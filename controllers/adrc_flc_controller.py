import numpy as np

from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManipulatorModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManipulatorModel(Tp, 1.0, 0.05)
        self.Kp = Kp
        self.Kd = Kd
        self.L = np.array(
            [
                [3 * p[0], 0],
                [0, 3 * p[1]],
                [3 * p[0] ** 2, 0],
                [0, 3 * p[1] ** 2],
                [p[0] ** 3, 0],
                [0, p[1] ** 3],
            ]
        )
        W = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        self.A = np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        self.B = np.zeros((6, 2))
        self.eso = ESO(self.A, self.B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        M = self.model.M(np.array([q[0], q[1], q_dot[0], q_dot[1]]))
        C = self.model.C(np.array([q[0], q[1], q_dot[0], q_dot[1]]))

        M = np.linalg.inv(M)
        M_C = -(M @ C)

        A = self.A.copy()
        B = self.B.copy()

        A[2, 2] = M_C[0, 0]
        A[2, 3] = M_C[0, 1]
        A[3, 2] = M_C[1, 0]
        A[3, 3] = M_C[1, 1]

        B[2, 0] = M[0, 0]
        B[2, 1] = M[0, 1]
        B[3, 0] = M[1, 0]
        B[3, 1] = M[1, 1]

        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q = x[:2]
        q_dot = x[2:]

        state = self.eso.get_state()
        q_hat = state[:4]
        q_dot_hat = state[4:]

        M = self.model.M([q_hat[0], q_hat[1], q_dot_hat[0], q_dot_hat[1]])
        C = self.model.C([q_hat[0], q_hat[1], q_dot_hat[0], q_dot_hat[1]])

        v = q_d_ddot + self.Kd @ (q_d_dot - q_dot) + self.Kp @ (q_d - q)
        u = M @ (v - state[4:]) + C @ q_dot_hat

        self.update_params(q_hat, q_dot_hat)
        self.eso.update(q, u)

        return u
