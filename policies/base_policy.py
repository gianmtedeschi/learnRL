"""Base policy class implementation"""

# imports
from abc import ABC, abstractmethod

# class
class BasePolicy(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.dim_state = None
        self.dim_action = None
        self.dim_params = None

    @abstractmethod
    def draw_action(self, state):
        pass

    @abstractmethod
    def set_parameters(self, thetas):
        pass

    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def compute_score(self, state, action):
        pass

    @abstractmethod
    def reduce_exploration(self):
        pass
