import numpy as np
import os
from typing import List, Optional, Tuple

from core.Agent.mdp_model import FiniteMDP

class Agent:
    def __init__(self, model: FiniteMDP, memory: Optional[str] = None):
        self.model = model
        if not self.model.verify_mdp():
            print("Model is invalid try passing another model")
        
        # Create weight_logs folder in core directory
        core_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_dir = os.path.join(core_dir, "weight_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        if memory:
            self.memory_path = os.path.join(self.log_dir, f"{memory}.npz")
            if os.path.exists(self.memory_path):
                data = np.load(self.memory_path)
                self.V = data['V']
                self.Q = data['Q']
                return
        else:
            # Create a unique name for a new log
            i = 0
            while os.path.exists(os.path.join(self.log_dir, f"log_{i}.npz")):
                i += 1
            self.memory_path = os.path.join(self.log_dir, f"log_{i}.npz")

        self.V: np.ndarray = np.zeros(model.n_states)
        self.Q: np.ndarray = np.zeros((self.model.n_states, self.model.n_actions))
        self.save_weights()

    def Bellman_operator_on_V(self, V: np.ndarray) -> np.ndarray:
        """
        Applies the Bellman optimality operator on V:
            TV(s) = max_a [ R(s,a) + gamma * sum_s' P(a,s,s') * V(s') ]
        
        Args:
            V: Value vector of shape (n_states,).
        Returns:
            Updated value vector of shape (n_states,).
        """
        P = self.model.P        # (A, S, S')
        R = self.model.R        # (S, A)
        gamma = self.model.gamma

        # For each (s,a): R(s,a) + gamma * sum_s' P(a,s,s') * V(s')
        # einsum 'asn,n->sa' contracts over s' to give shape (S, A)
        Q_sa = R + gamma * np.einsum('asn,n->sa', P, V)

        # TV(s) = max_a Q(s,a)
        return np.max(Q_sa, axis=1)

    def Bellman_operator_on_Q(self, Q: np.ndarray) -> np.ndarray:
        """
        Applies the Bellman optimality operator on Q:
            TQ(s,a) = R(s,a) + gamma * sum_s' P(a,s,s') * max_a' Q(s',a')
        
        Args:
            Q: Action-value matrix of shape (n_states, n_actions).
        Returns:
            Updated action-value matrix of shape (n_states, n_actions).
        """
        P = self.model.P        # (A, S, S')
        R = self.model.R        # (S, A)
        gamma = self.model.gamma

        # max_a' Q(s', a') for each s' → shape (n_states,)
        V_from_Q = np.max(Q, axis=1)

        # sum_s' P(a,s,s') * V(s') for each (s,a) → shape (S, A)
        # einsum 'asn,n->sa' contracts over s'
        TQ = R + gamma * np.einsum('asn,n->sa', P, V_from_Q)

        return TQ

    def value_iteration_on_V(self, tol: float = 1e-10, max_iter: int = 10000) -> tuple:
        """
        Runs value iteration using the Bellman operator on V until convergence.
        
        Args:
            tol: Convergence threshold on ||V_new - V_old||_inf.
            max_iter: Maximum number of iterations.
        Returns:
            (V, n_iters) — converged value function and number of iterations.
        """
        V = np.copy(self.V)
        for i in range(1, max_iter + 1):
            V_new = self.Bellman_operator_on_V(V)
            if np.max(np.abs(V_new - V)) < tol:
                self.V = V_new
                self.Q = self.model.R + self.model.gamma * np.einsum('asn,n->sa', self.model.P, V_new)
                self.save_weights()
                return V_new, i
            V = V_new
        
        self.V = V
        self.Q = self.model.R + self.model.gamma * np.einsum('asn,n->sa', self.model.P, V)
        self.save_weights()
        print(f"Warning: Value iteration on V did not converge in {max_iter} iterations.")
        return V, max_iter

    def value_iteration_on_Q(self, tol: float = 1e-10, max_iter: int = 10000) -> tuple:
        """
        Runs value iteration using the Bellman operator on Q until convergence.
        
        Args:
            tol: Convergence threshold on ||Q_new - Q_old||_inf.
            max_iter: Maximum number of iterations.
        Returns:
            (Q, n_iters) — converged action-value function and number of iterations.
        """
        Q = np.copy(self.Q)
        for i in range(1, max_iter + 1):
            Q_new = self.Bellman_operator_on_Q(Q)
            if np.max(np.abs(Q_new - Q)) < tol:
                self.Q = Q_new
                self.V = np.max(Q_new, axis=1)
                self.save_weights()
                return Q_new, i
            Q = Q_new
        
        self.Q = Q
        self.V = np.max(Q, axis=1)
        self.save_weights()
        print(f"Warning: Value iteration on Q did not converge in {max_iter} iterations.")
        return Q, max_iter

    def greedy_path(self, water: bool = True, max_steps: int = 25) -> List[Tuple[int, int]]:
        """
        Generate a path by greedily following the optimal value function V.

        - water=True  → start at Lake (0,0) with water, goal is Fire (4,4).
                         Uses states 25-49 (water phase w=1).
        - water=False → start at Fire (4,4) without water (already delivered),
                         goal is Lake (0,0). Uses states 0-24 (phase w=0).

        At each step, looks at the 4 cardinal neighbors, picks the one with
        the highest V value, and moves there. Terminates when the goal is
        reached or after max_steps (with a warning).

        Returns:
            path: List of (x, y) grid coordinates visited.
        """
        actions = [(0, 1), (0, -1), (1, 0), (-1, 0) , (0,0)]  # N, S, E, W,Hower

        if water:
            start, goal, w = (0, 0), (4, 4), 1
        else:
            start, goal, w = (4, 4), (0, 0), 0

        x, y = start
        path = [(x, y)]

        for step in range(max_steps):
            if (x, y) == goal:
                break

            best_val = -np.inf
            best_pos = (x, y)
            for dx, dy in actions:
                nx, ny = x + dx, y + dy
                if (nx,ny) ==  goal:
                    path.append((nx,ny))
                    return path
            
                if 0 <= nx < 5 and 0 <= ny < 5:
                    s_idx = nx + 5 * ny + 25 * w
                    print("s_idx = ",s_idx," V[s_idx] = ",self.V[s_idx]," nx = ",nx," ny = ",ny)
                    if self.V[s_idx] > best_val:
                        best_val = self.V[s_idx]
                        best_pos = (nx, ny)

            x, y = best_pos
            path.append((x, y))
        else:
            if (x, y) != goal:
                print(f"Warning: Path did not reach goal {goal} after {max_steps} steps. Terminating.")

        return path

    def save_weights(self):
        """Utility to persist V and Q matrices."""
        np.savez(self.memory_path, V=self.V, Q=self.Q)