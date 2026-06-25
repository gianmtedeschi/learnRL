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
            deterministic: bool = False,
            linear: bool = False,
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

        # self.history = history
        self.history = BinaryTree()
        
        # Per-leaf parameter count. A constant leaf holds an action-sized mean
        # (dim_action); a linear leaf holds a full gain matrix of shape
        # (dim_action, dim_state), stored flattened -> dim_action * dim_state.
        self.tot_params = dim_action * dim_state if linear else dim_action
        self.deterministic = deterministic
        self.linear = linear

        return

    def _leaf_mean(self, theta, state) -> np.array:
        """Action mean for one leaf parameter `theta` at `state`.

        Constant leaves: the mean is the (action-sized) parameter itself.
        Linear leaves: theta is a flattened (dim_action, dim_state) gain and the
        mean is `theta @ state` (the LQ-style linear controller a = K s).
        """
        theta = np.asarray(theta, dtype=np.float64)
        if self.linear:
            gain = theta.reshape(self.dim_action, self.dim_state)
            return gain @ np.ravel(state)
        return np.ravel(theta)

    def draw_action(self, state) -> float:
        state = np.ravel(state)

        leaf = self.history.find_region_leaf(state, policy=True)
        theta = self.history.root.val[0] if leaf is None else leaf.val[0]

        mean = self._leaf_mean(theta, state)
        action = np.random.normal(mean, self.std_dev, size=self.dim_action)

        return np.ravel(action)

    def compute_score(self, state, action) -> np.array:
        if self.std_dev == 0:
            return super().compute_score(state, action)

        state = np.ravel(state)
        action = np.ravel(action)

        leaves = self.history.get_all_leaves()
        scores = np.zeros((len(leaves), self.tot_params))

        leaf = self.history.find_region_leaf(state, policy=True)

        # Write the score only at the region the state actually falls in. Match
        # by unique node_id (identity), not by parameter value: distinct leaves
        # can hold equal values, and value-matching would (wrongly) credit the
        # score to every such leaf.
        for position, node in enumerate(leaves):
            if node.node_id == leaf.node_id:
                deviation = action - self._leaf_mean(leaf.val[0], state)
                if self.linear:
                    # grad wrt a (dim_action, dim_state) gain: outer(dev, state),
                    # flattened to match the leaf's flattened parameter layout.
                    scores[position] = np.outer(deviation, state).ravel() / (self.std_dev ** 2)
                else:
                    scores[position] = deviation / (self.std_dev ** 2)
                break

        return scores
    
    def reduce_exploration(self):
        self.std_dev = np.clip(self.std_dev - self.std_decay, self.std_min, np.inf)


