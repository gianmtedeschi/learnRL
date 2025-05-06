"""Trajectory sampler"""

# imports
from envs import BaseEnv
from policies import BasePolicy
from data_processors import BaseProcessor
import numpy as np
import math
import copy
from common.utils import *
from joblib import Parallel, delayed

# worker method for parallel computation
def planning_sampling_worker(
        env: BaseEnv = None,
        pol: BasePolicy = None,
        dp: BaseProcessor = None,
        params: np.ndarray = None,
        starting_state: np.ndarray = None,
        planning_horizon: int = 1,
        persistence: bool = False,
        seed: int = 0
) -> list:
    """Worker collecting a single trajectory.

    Args:
        env (BaseEnv, optional): the env to employ. Defaults to None.
        
        pol (BasePolicy, optional): the policy to play. Defaults to None.
        
        dp (Baseprocessor, optional): the data processor to employ. 
        Defaults to None.
        
        params (np.array, optional): the parameters to plug into the policy. 
        Defaults to None.
        
        starting_state (np.array, optional): the state to which the env should 
        be initialized. Defaults to None.

    Returns:
        list: [performance, reward, scores]
    """
    trajectory_sampler = TrajectorySampler(env=env, pol=pol, data_processor=dp)
    res = trajectory_sampler.collect_trajectory_mixed_planning(params=params, starting_state=starting_state, planning_horizon=planning_horizon, persistence=persistence, seed=seed)
    
    return res

def pg_sampling_worker(
        env: BaseEnv = None,
        pol: BasePolicy = None,
        dp: BaseProcessor = None,
        params: np.ndarray = None,
        starting_state: np.ndarray = None,
        seed: int = 0
) -> list:
    """Worker collecting a single trajectory.

    Args:
        env (BaseEnv, optional): the env to employ. Defaults to None.
        
        pol (BasePolicy, optional): the policy to play. Defaults to None.
        
        dp (Baseprocessor, optional): the data processor to employ. 
        Defaults to None.
        
        params (np.array, optional): the parameters to plug into the policy. 
        Defaults to None.
        
        starting_state (np.array, optional): the state to which the env should 
        be initialized. Defaults to None.

    Returns:
        list: [performance, reward, scores]
    """
    trajectory_sampler = TrajectorySampler(env=env, pol=pol, data_processor=dp)
    res = trajectory_sampler.collect_trajectory(params=params, starting_state=starting_state, seed=seed)
    
    return res

def pg_sampling_worker_bpo(
        env: BaseEnv = None,
        pol: BasePolicy = None,
        pol_b : BasePolicy = None,
        dp: BaseProcessor = None,
        params: np.ndarray = None,
        params_b: np.ndarray = None,
        starting_state: np.ndarray = None,
        seed: int = 0
) -> list:
    """Worker collecting a single trajectory.

    Args:
        env (BaseEnv, optional): the env to employ. Defaults to None.
        
        pol (BasePolicy, optional): the policy to play. Defaults to None.
        
        dp (Baseprocessor, optional): the data processor to employ. 
        Defaults to None.
        
        params (np.array, optional): the parameters to plug into the policy. 
        Defaults to None.
        
        starting_state (np.array, optional): the state to which the env should 
        be initialized. Defaults to None.

    Returns:
        list: [performance, reward, scores]
    """
    trajectory_sampler = TrajectorySampler(env=env, pol=pol, pol_b = pol_b, data_processor=dp)
    res = trajectory_sampler.collect_trajectory_forBPO(params_target = params, params_behavioural = params_b, starting_state=starting_state, seed=seed)
    
    return res

# sampler class for action-based methods
class TrajectorySampler:
    def __init__(
            self, env: BaseEnv = None,
            pol: BasePolicy = None,
            pol_b: BasePolicy = None,
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
        
        self.pol_b = pol_b

        return

    def collect_trajectory(
            self, params: np.array = None, starting_state=None, split=False, seed=0
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
        self.env.reset(seed=seed)
        if starting_state is not None:
            self.env.state = copy.deepcopy(starting_state)

        # initialize parameters
        np.random.seed(seed)
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
            state, rew, done, _ = self.env.step(a)

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
        
        if split:
            return [perf, rewards, scores, states]
        else:  
            return [perf, rewards, scores]
        
    def collect_trajectory_forBPO(
            self, params_target: np.array = None,
            starting_state=None, split=False, seed=0,
            params_behavioural: np.array = None
            
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
        self.env.reset(seed=seed)
        if starting_state is not None:
            self.env.state = copy.deepcopy(starting_state)

        # initialize parameters
        np.random.seed(seed)
        perf = 0
        rewards = np.zeros(self.env.horizon, dtype=np.float64)
        if split:
            scores = np.zeros((self.env.horizon, len(self.pol.history.get_all_leaves()), self.pol.tot_params), dtype=np.float64)
        else:
            scores = np.zeros((self.env.horizon, self.pol.tot_params), dtype=np.float64)

        states = np.zeros((self.env.horizon, self.env.state_dim), dtype=np.float64)
        actions = np.zeros((self.env.horizon, self.env.action_dim), dtype= np.float64)
        logprobs_t = np.zeros(self.env.horizon, dtype = np.float64)
        logprobs_b = np.zeros(self.env.horizon, dtype = np.float64)
        if params_target is not None:
            self.pol.set_parameters(thetas=copy.deepcopy(params_target))
        if params_behavioural is not None:
            self.pol_b.set_parameters(thetas = copy.deepcopy(params_behavioural))

        # act
        for t in range(self.env.horizon):
            # retrieve the state
            state = self.env.state

            # transform the state
            features = self.dp.transform(state=state)

            # select the action
            a = self.pol_b.draw_action(state=features)
            score = self.pol_b.compute_score(state=features, action=a)
            
            logprob_t = self.pol.compute_logprob(features, a)
            logprob_b = self.pol_b.compute_logprob(features, a)

            # play the action
            state, rew, done, _ = self.env.step(a)

            # update the performance index
            perf += (self.env.gamma ** t) * rew

            # update the vectors of rewards scores and state
            rewards[t] = rew
            scores[t, :] = score
            states[t, :] = state
            actions[t,:] = a
            logprobs_t[t] = logprob_t
            logprobs_b[t] = logprob_b

            if done:
                if t < self.env.horizon - 1:
                    rewards[t+1:] = 0
                    scores[t+1:] = 0
                    logprobs_t[t+1:] = 0
                    logprobs_b[t+1:] = 0
                    # dovrei gestire le logprob in qualche moddo, tipo -inf ? però poi quando li sommo è un problema, ha più senso 0
                    
                break
        
        if split:
            return [perf, rewards, scores, states, logprobs_t, logprobs_b]
        else:  
            return [perf, rewards, scores, states, logprobs_t, logprobs_b, actions]
    
    
    def collect_trajectory_mixed_planning(
            self, params: np.array = None, starting_state=None, planning_horizon=1, persistence=False, seed=0
    ) -> list:
        """
        Summary:
            Function collecting a trajectory reward for a particular theta
            configuration.
        Args:
            params (np.array): the current sampling of theta values
            starting_state (any): the starting state for the iterations
            planning_horizon: the horizon of planning 
        Returns:
            list of:
                float: the discounted reward of the trajectory
                np.array: vector of all the rewards
                np.array: vector of all the scores
        """
        # reset the environment
        self.env.reset(seed=seed)
        if starting_state is not None:
            self.env.state = copy.deepcopy(starting_state)

        # initialize parameters
        np.random.seed(seed)
        perf = 0
        rewards = np.zeros(self.env.horizon, dtype=np.float64)
        scores = np.zeros((self.env.horizon, self.pol.tot_params), dtype=np.float64)

        if params is not None:
            self.pol.set_parameters(thetas=copy.deepcopy(params))

        state = self.env.state
        # act
        for t in range(math.ceil(self.env.horizon/planning_horizon)):
            # retrieve the state
            # state = self.env.state

            # transform the state
            features = self.dp.transform(state=state)
            
            # sample the sequence of actions
            action = self.pol.draw_action(state=features)
                        
            # compute the score
            score = self.pol.compute_score(state=features, action=action)

            if persistence:
                # repeat the action for the planning horizon
                action = np.tile(action, planning_horizon).ravel()

            # reshape the action according to the planning horizon
            action = np.array(np.split(action, planning_horizon))

            seq_reward = .0
            for i, a in enumerate(action):
                # play the action
                state, rew, done, _ = self.env.step(action=a)
                seq_reward += (self.env.gamma ** i) * rew
                # noisy state
                # state += np.random.normal(loc=0, scale=0.5, size=self.env.state_dim)
                if done:
                    break

            # update the performance index
            perf += (self.env.gamma ** (t * planning_horizon)) * seq_reward

            # update the vectors of rewards scores and state
            rewards[t] = seq_reward
            scores[t, :] = score

            if done:
                if t < self.env.horizon - 1:
                    rewards[t+1:] = 0
                    scores[t+1:] = 0
                break

        return [perf, rewards, scores]


# sampler class for parameter-based method
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
