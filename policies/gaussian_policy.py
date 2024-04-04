"""Implementation of a Gaussian Policy"""

# imports
from policies import BasePolicy
from abc import ABC
import numpy as np
import copy


# class
class GaussianPolicy(BasePolicy, ABC):
    """
    Implementation of a Gaussian Policy.
    In case of linear policy the mean will be: parameters @ state.
    The standard deviation is fixed and is defined by the user.
    """
    def __init__(
            self, parameters: np.array = None,
            std_dev: float = 0.1,
            std_decay: float = 0,
            std_min: float = 1e-4,
            dim_state: int = 1,
            dim_action: int = 1,
            multi_linear: bool = False,
            constant: bool = False

    ) -> None:
        # Superclass initialization
        super().__init__()

        # Attributes with checks
        err_msg = "[GaussPolicy] parameters is None!"
        assert parameters is not None, err_msg
        self.parameters = parameters

        err_msg = "[GaussPolicy] standard deviation is negative!"
        assert std_dev > 0, err_msg
        self.std_dev = std_dev

        # Additional attributes
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.tot_params = dim_action * dim_state
        self.multi_linear = multi_linear
        self.constant = constant
        self.std_decay = std_decay
        self.std_min = std_min

        return

    def draw_action(self, state) -> float:
        if state.size != self.dim_state:
            err_msg = "[GaussPolicy] the state has not the same dimension of the parameter vector:"
            err_msg += f"\n{len(state)} vs. {self.dim_state}"
            raise ValueError(err_msg)

        if state.size == 1:
            state = [state]

        if self.constant:
            mean = self.parameters
        else:
            mean = np.array(self.parameters @ state, dtype=np.float64)

        action = np.array(np.random.normal(mean, self.std_dev), dtype=np.float64)
        return action

    def reduce_exploration(self):
        self.std_dev = np.clip(self.std_dev - self.std_decay, self.std_min, np.inf)

    def set_parameters(self, thetas) -> None:
        if not self.multi_linear:
            self.parameters = copy.deepcopy(thetas)
        else:
            self.parameters = np.array(np.split(thetas, self.dim_action))

    def compute_score(self, state, action) -> np.array:
        if self.std_dev == 0:
            return super().compute_score(state, action)

        state = np.ravel(state)
        action_deviation = action - (self.parameters @ state)
        if self.multi_linear:
            state = np.tile(state, self.dim_action).reshape((self.dim_action, self.dim_state))
            action_deviation = action_deviation[:, np.newaxis]

        scores = (action_deviation * state) / (self.std_dev ** 2)

        if self.multi_linear:
            scores = np.ravel(scores)
        if self.constant:
            scores = (action - self.parameters) / (self.std_dev ** 2)
        return scores
