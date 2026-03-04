import numpy as np
from typing import List, Optional

class FiniteMDP:
    """
    A class representing a Finite Markov Decision Process (S, A, P, R, gamma).
    """
    def __init__(self):
        self.n_states: int = -1
        self.n_actions: int = -1
        self.gamma: float = -1
        self.S: List[str] = []
        self.A: List[tuple] = []
        self.P: Optional[np.ndarray] = None  # 3D: (A, S, S')
        self.R: Optional[np.ndarray] = None  # 2D: (S, A)

    def is_valid(self, x: int, y: int) -> bool:
        """Checks if (x, y) is within the 5x5 grid."""
        return 0 <= x < 5 and 0 <= y < 5

    def _update_probabilities_n_rewards_of_state(self, s_idx: int):
        """
        Helper to fill transition probabilities and rewards using the state label.
        """
        pos_idx = s_idx % 25
        x, y = pos_idx % 5, pos_idx // 5
        w = s_idx // 25
        label = self.S[s_idx]

        # 1. Terminal State Check: Boulder or Successful delivery to Fire zone
        # Absorbing states with R=0 so V=0; the crash/success penalty is
        # captured in R(neighbor, a) when transitioning INTO these states.
        if "Boulder" in label:
            for a_idx in range(self.n_actions):
                self.P[a_idx, s_idx, s_idx] = 1.0
                self.R[s_idx, a_idx] = 0
            return

        if "Fire" in label and w == 1:
            for a_idx in range(self.n_actions):
                self.P[a_idx, s_idx, s_idx] = 1.0
                self.R[s_idx, a_idx] = 0.0
            return

        is_smoke = "Smoke" in label
        lake_pos = (0, 0)
        
        temp = 0
        for a_idx, action in enumerate(self.A):
            # 2. Hover Action: Deterministic stay (unaffected by wind)
            if a_idx == 4: # Hover action
                nx, ny = x, y
                nw = 1 if (nx, ny) == lake_pos else w
                s_next = nx + 5*ny + 25*nw
                self.P[a_idx, s_idx, s_next] = 1.0
                
                # Reward for hover
                r_hover = -1.0
                if "Smoke" in label: r_hover -= 90.0 # Entering hazard additional penalty
                #if "Lake" in label and w == 0: r_hover = 100.0 # Picking up water
                self.R[s_idx, a_idx] = r_hover
                """
                # Hover: always -1. Description says "entering hazardous regions"
                # incurs -10. Hovering means STAYING, not entering, so no extra penalty.
                self.R[s_idx, a_idx] = -1.0
                """
                continue

            # 3. Movement Actions: Stochastic wind effects
            p_intended, p_stay, p_perp = (0.4, 0.4, 0.1) if is_smoke else (0.7, 0.1, 0.1)
            
            perpendiculars = [(1, 0), (-1, 0)] if action[0] == 0 else [(0, 1), (0, -1)]
            
            outcomes = [
                (action, p_intended),
                ((0, 0), p_stay),
                (perpendiculars[0], p_perp),
                (perpendiculars[1], p_perp)
            ]

            expected_reward = 0.0
            for move, prob in outcomes:
                nx, ny = x + move[0], y + move[1]
                
                # Boundary check: If move goes off-grid, stay in current cell
                if not self.is_valid(nx, ny):
                    nx, ny = x, y
                
                # Update water status if entering Lake
                nw = 1 if (nx, ny) == lake_pos else w
                s_prime = nx + 5*ny + 25*nw
                
                self.P[a_idx, s_idx, s_prime] += prob

                # Calculate reward for this outcome
                next_label = self.S[s_prime]
                if "Boulder" in next_label:
                    r_out = -100.0
                elif "Fire" in next_label and nw == 1:
                    r_out = 100.0
                #elif "Lake" in next_label and w == 0:
                    #r_out = 100
                else:
                    r_out = -1.0 # Per-step penalty
                    if "Smoke" in next_label: r_out -= 90.0 # Additional smoke penalty
                
                expected_reward += prob * r_out
            
            self.R[s_idx, a_idx] = expected_reward

    def initialization_MDP_with_DA6400_Question1_manner(self,gamma) -> tuple:
        # Define Environment Special Locations
        lake = (0, 0)
        smoke = {(1, 2), (3, 2)}
        boulders = {(2, 4), (3, 4)}
        fire = (4, 4)

        # 1. Initialize State Labels
        self.n_states = 50
        self.S = [""] * self.n_states
        for w in [0, 1]:
            for y in range(5):
                for x in range(5):
                    idx = x + 5*y + 25*w
                    pos = (x, y)
                    label = "Free"
                    if pos == lake: label = "Lake"
                    elif pos == fire: label = "Fire"
                    elif pos in smoke: label = "Smoke"
                    elif pos in boulders: label = "Boulder"
                    
                    status = "Water" if w else "Empty"
                    self.S[idx] = f"{label}_{status}"

        # 2. Initialize Actions (N, S, E, W, Hover)
        self.A = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
        self.n_actions = len(self.A)
        self.gamma = gamma
        
        # 3. Initialize Matrices
        self.P = np.zeros((self.n_actions, self.n_states, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions))

        # 4. Fill Matrices using helper
        for s_idx in range(self.n_states):
            self._update_probabilities_n_rewards_of_state(s_idx)

        return self.S, self.A, self.P, self.R, self.gamma

    def initialization_MDP_with_custom_S_A_P_R_gamma(self, S, A, P, R, gamma):
        self.S = S
        self.A = A
        self.P = P
        self.R = R  
        self.gamma = gamma
        self.n_actions = len(self.A)
        self.n_states = len(self.S)

    def verify_mdp(self) -> bool:
        """
        Checks:
          1. All MDP components are properly initialized.
          2. All transition probabilities sum to 1.
        """
        # --- Step 1: Initialization Check ---
        init_errors = []
        if self.n_states == -1:  init_errors.append("n_states is not set (-1)")
        if self.n_actions == -1: init_errors.append("n_actions is not set (-1)")
        if self.gamma == -1:     init_errors.append("gamma is not set (-1)")
        if len(self.S) == 0:     init_errors.append("S (state list) is empty")
        if len(self.A) == 0:     init_errors.append("A (action list) is empty")
        if self.P is None:       init_errors.append("P (transition matrix) is None")
        if self.R is None:       init_errors.append("R (reward matrix) is None")

        if init_errors:
            print("MDP is NOT initialized. Issues found:")
            for e in init_errors: print(f"  - {e}")
            return False

        # --- Step 2: Probability Sum Check ---
        all_valid = True
        for a in range(self.n_actions):
            for s in range(self.n_states):
                prob_sum = np.sum(self.P[a, s, :])
                if not np.isclose(prob_sum, 1.0) and not np.isclose(prob_sum, 0.0):
                    print(f"  Warning: State {s} ({self.S[s]}), Action {a} sums to {prob_sum:.4f}")
                    all_valid = False

        if all_valid:
            print("MDP verified: All components initialized and probabilities sum to 1.0")
        return all_valid
