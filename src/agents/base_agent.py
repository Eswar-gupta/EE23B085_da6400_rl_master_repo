"""
Base agent interface.

All RL agents should inherit from BaseAgent and implement:
  - select_action(state) -> action
  - update(transition)   -> None
  - save(path)           -> None
  - load(path)           -> None
"""


class BaseAgent:
    """Abstract base class for all RL agents."""

    def select_action(self, state):
        """Return an action given the current state."""
        raise NotImplementedError

    def update(self, transition):
        """Update the agent given a (s, a, r, s', done) transition."""
        raise NotImplementedError

    def save(self, path: str):
        """Persist model weights / parameters to *path*."""
        raise NotImplementedError

    def load(self, path: str):
        """Load model weights / parameters from *path*."""
        raise NotImplementedError
