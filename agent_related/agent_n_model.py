import numpy as np
from typing import List, Tuple

class Agent:
    def __init__(self, S: List[str], A: List[tuple], P: np.ndarray, R: np.ndarray, gamma: float):
        self.S = S
        self.A = A
        self.P = P
        self.R = R
        self.gamma = gamma
        self.n_states = len(S)
        self.n_actions = len(A)

        for a in range(self.n_actions):
            for s in range(self.n_states):
                prob_sum = np.sum(self.P[a, s, :])
                assert np.isclose(prob_sum, 1.0), f"P[action={a}, state={s}] sums to {prob_sum}"

    def bellman_operator(self, P: np.ndarray, R: np.ndarray, V: np.ndarray, tol: float = 1e-10) -> Tuple[np.ndarray, bool]:
        Q = R + self.gamma * np.einsum('asn,n->sa', P, V)
        V_new = np.max(Q, axis=1)
        saturated = np.max(np.abs(V_new - V)) < tol
        return V_new, saturated

    def compute_Q(self, P: np.ndarray, R: np.ndarray, V: np.ndarray) -> np.ndarray:
        return R + self.gamma * np.einsum('asn,n->sa', P, V)
