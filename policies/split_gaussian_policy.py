"""Implementation of a Gaussian Policy for the Split PG"""

# imports
from policies import BasePolicy
from policies import GaussianPolicy
import numpy as np
import copy
from common.tree import BinaryTree


# class
class SplitGaussianPolicy(GaussianPolicy, BasePolicy):
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
            constant: bool = True,
            history: BinaryTree = BinaryTree()

    ) -> None:
        # Superclass initialization
        super().__init__(parameters, std_dev)

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

        self.history = history
        return

    def draw_action(self, state) -> float:
        if state.size != self.dim_state:
            err_msg = "[GaussPolicy] the state has not the same dimension of the parameter vector:"
            err_msg += f"\n{len(state)} vs. {self.dim_state}"
            raise ValueError(err_msg)

        if self.history is None:
            mean = self.parameters
            action = np.array(np.random.normal(mean, self.std_dev), dtype=np.float64)

        mean = self.history.find_closest_leaf(state.item())
        if mean is None:
            action = np.random.normal(self.history.root.val[0], np.identity(1) * self.std_dev)
        else:
            action = np.random.normal(mean.val[0], np.identity(1) * self.std_dev)

        return action

    def compute_score(self, state, action) -> np.array:
        if self.std_dev == 0:
            return super().compute_score(state, action)

        leaf = self.history.find_closest_leaf(state.item())
        scores = (action - leaf.val[0]) / (self.std_dev ** 2)

        return scores
