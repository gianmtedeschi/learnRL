"""Trajectory sampler"""

# imports
from envs import BaseEnv
from policies import BasePolicy
from data_processors import BaseProcessor
import numpy as np
import copy

# class
class TrajectorySampler:
    def __init__(
            self, env: BaseEnv = None,
            pol: BasePolicy = None,
            data_processor: BaseProcessor = None
    ) -> None:
        err_msg = "[PGTrajectorySampler] no environment provided!"
        assert env is not None, err_msg
        self.env = env

        err_msg = "[PGTrajectorySampler] no policy provided!"
        assert pol is not None, err_msg
        self.pol = pol

        err_msg = "[PGTrajectorySampler] no data_processor provided!"
        assert data_processor is not None, err_msg
        self.dp = data_processor

        return

    def collect_trajectory(
            self, params: np.array = None, starting_state=None, split=False
    ) -> list:
        """
        Summary:
            Function collecting a trajectory reward for a particular theta
            configuration.
        Args:
            params (np.array): the current sampling of theta values
            starting_state (any): teh starting state for the iterations
        Returns:
            list of:
                float: the discounted reward of the trajectory
                np.array: vector of all the rewards
                np.array: vector of all the scores
        """
        # reset the environment
        self.env.reset()
        if starting_state is not None:
            self.env.state = copy.deepcopy(starting_state)

        # initialize parameters
        np.random.seed()
        perf = 0
        rewards = np.zeros(self.env.horizon, dtype=np.float64)
        if split:
            scores = np.zeros((self.env.horizon, len(self.pol.history.get_all_leaves()), self.pol.tot_params), dtype=np.float64)
        else:
            scores = np.zeros((self.env.horizon, self.pol.tot_params), dtype=np.float64)

        states = np.zeros((self.env.horizon, self.env.state_dim), dtype=np.float64)
        if params is not None:
            self.pol.set_parameters(thetas=copy.deepcopy(params))

        

        # act
        for t in range(self.env.horizon):
            # retrieve the state
            state = self.env.state

            # transform the state
            features = self.dp.transform(state=state)

            # select the action
            a = self.pol.draw_action(state=features)
            score = self.pol.compute_score(state=features, action=a)

            # play the action
            state, rew, done, _ = self.env.step(action=a)

            # update the performance index
            perf += (self.env.gamma ** t) * rew

            # update the vectors of rewards scores and state
            rewards[t] = rew
            scores[t, :] = score
            states[t, :] = np.ravel(state)

            if done:
                if t < self.env.horizon - 1:
                    rewards[t+1:] = 0
                    scores[t+1:] = 0
                break

        return [perf, rewards, scores, states]
