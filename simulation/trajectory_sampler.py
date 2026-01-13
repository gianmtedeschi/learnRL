"""Trajectory sampler"""

# imports
from envs import BaseEnv
from policies import BasePolicy
from data_processors import BaseProcessor
import numpy as np
import copy
import joblib
from common.utils import *

# class
class TrajectorySampler:
    def __init__(
            self, env: BaseEnv = None,
            pol: BasePolicy = None,
            data_processor: BaseProcessor = None,
            n_jobs : int = 1
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
        self.n_jobs = n_jobs


        return

    def collect_trajectories(self, params: np.array = None, starting_state=None, split=False, num_trajectories=1) -> list:
        """
        Collect multiple trajectories in parallel.

        Args:
            params (np.array): the current sampling of theta values
            starting_state (any): the starting state for the iterations
            split (bool): whether to use split scores
            num_trajectories (int): number of trajectories to collect

        Returns:
            list: List of collected trajectories
        """
        results = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(self.collect_trajectory)(params, starting_state, split)
            for _ in range(num_trajectories)
        )
        return results

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
            states[t, :] = state

            if done:
                if t < self.env.horizon - 1:
                    rewards[t+1:] = 0
                    scores[t+1:] = 0
                break

        return [perf, rewards, scores, states]


class ParameterSampler:
    """Sampler for PGPE."""
    def __init__(
            self, env: BaseEnv = None,
            pol: BasePolicy = None,
            data_processor: BaseProcessor = None,
            episodes_per_theta: int = 1
    ) -> None:
        """
        Summary:
            Initialization.

        Args:
            env (BaseEnv, optional): the env to employ. Defaults to None.
            
            pol (BasePolicy, optional): the poliy to play. Defaults to None.
            
            data_processor (BaseProcessor, optional): the data processor to use. 
            Defaults to None.
            
            episodes_per_theta (int, optional): how many trajectories to 
            evaluate for each sampled theta. Defaults to 1.
            
            n_jobs (int, optional): how many theta sample (and evaluate) 
            in parallel. Defaults to 1.
        """
        err_msg = "[PGPETrajectorySampler] no environment provided!"
        assert env is not None, err_msg
        self.env = env

        err_msg = "[PGPETrajectorySampler] no policy provided!"
        assert pol is not None, err_msg
        self.pol = pol

        err_msg = "[PGPETrajectorySampler] no data_processor provided!"
        assert data_processor is not None, err_msg
        self.dp = data_processor

        self.episodes_per_theta = episodes_per_theta
        self.trajectory_sampler = TrajectorySampler(
            env=self.env,
            pol=self.pol,
            data_processor=self.dp
        )

        return

    def collect_trajectory(self, params: np.array, gaps=True) -> list:
        """
        Summary:
            Collect the trajectories for a sampled parameter configurations.

        Args:
            params (np.array): hyper-policy configuration.

        Returns:
            list: [params, performance]
        """
        # sample a parameter configuration
        dim = len(params)
        thetas = np.zeros(dim, dtype=np.float64)

        # if we are not using gaps sample for pgpe
        if not gaps:
            thetas = np.random.normal(
                params[RhoElem.MEAN], RhoElem.STD)

        # collect performances over the sampled parameter configuration
        raw_res = []
        for i in range(100):
            if gaps:
                thetas = np.random.normal(params, self.pol.std_dev)
            raw_res.append(self.trajectory_sampler.collect_trajectory(
                params=thetas, starting_state=None)
            )

        # extract the results
        perf_res = np.zeros(100, dtype=np.float64)

        if gaps:
            scores = np.zeros((self.env.horizon, len(self.pol.history.get_all_leaves()), self.pol.tot_params), dtype=np.float64)
        else:
            scores = np.zeros((self.env.horizon, self.pol.tot_params), dtype=np.float64)

        states = np.zeros((self.env.horizon, self.env.state_dim), dtype=np.float64)

        for i, elem in enumerate(raw_res):
            # perf_res[i] = elem[TrajectoryResults.PERF]
            perf_res = elem[TrajectoryResults.PERF]
            scores[i] = elem[TrajectoryResults.ScoreList]
            states[i] = elem[TrajectoryResults.StateList]

        return [perf_res, scores, states]
