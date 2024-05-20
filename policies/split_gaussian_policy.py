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
            history: BinaryTree = BinaryTree()

    ) -> None:
        # Superclass initialization
        super().__init__(parameters, std_dev)

        # Attributes with checks
        err_msg = "[TreePolicy] parameters is None!"
        assert parameters is not None, err_msg
        self.parameters = parameters

        err_msg = "[TreePolicy] standard deviation is negative!"
        assert std_dev > 0, err_msg
        self.std_dev = std_dev

        # Additional attributes
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.std_decay = std_decay
        self.std_min = std_min

        self.history = history
        self.tot_params = dim_action
        return

    def draw_action(self, state) -> float:
        if self.history is None:
            mean = self.parameters
            action = np.array(np.random.normal(mean, self.std_dev), dtype=np.float64)

        mean = self.history.find_region_leaf(state, policy=True)
        if mean is None:
            action = np.random.normal(self.history.root.val[0], np.identity(1) * self.std_dev)
        else:
            action = np.random.normal(mean.val[0], np.identity(1) * self.std_dev)

        return action.ravel()

    def compute_score(self, state, action) -> np.array:
        if self.std_dev == 0:
            return super().compute_score(state, action)
        
        scores = np.zeros((len(self.history.get_all_leaves()), self.tot_params))
        
        # scores = np.zeros(self.tot_params)
        leaf = self.history.find_region_leaf(state, policy=True)

        for position, Node in enumerate(self.history.get_all_leaves()):
            if np.all(leaf.val[0] == Node.val[0]):
                #TODO np.ravel
                scores[position] = (action - leaf.val[0]) / (self.std_dev ** 2)
        
        return scores
    
    def reduce_exploration(self):
        self.std_dev = np.clip(self.std_dev - self.std_decay, self.std_min, np.inf)