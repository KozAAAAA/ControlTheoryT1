from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        z = self.state[: np.newaxis]
        z_dot = self.A @ z + self.B @ u + (self.L @ (q - self.W @ z))
        self.state = (z + z_dot * self.Tp).squeeze()

    def get_state(self):
        return self.state
