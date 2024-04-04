"""Adaptive Policy Gradient Implementation

structure idea --> use this class solely for split logic
                    learning part performed by policy_gradient.py
                    this class should return a new costant gaussian policy

new policy structure!!

pol = GaussianPolicy(
    parameters=thetas_split,
    dim_state=self.env.ds,
    dim_action=self.env.da,
    std_dev=0.1,
    std_decay=0,
    std_min=1e-6,
    multi_linear=False,
    constant=True
)

todo --> specialize gaussian policy for the use case
        split point should be part of the policy since are needed for draw_action
        and compute_score (play using param in your region, computer score in your region)

todo --> adjust the logic of the algorithm, check the sign of the current gradient and the
        previous one, when they are opposite activate the split process, before that just normally
        update the policy

todo --> from split return also the split point associated to the possible division as to be able
        to understand who is the father of the 2 params
"""



# imports
import numpy as np
from envs.base_env import BaseEnv
from policies import BasePolicy, GaussianPolicy, SplitGaussianPolicy
from data_processors import BaseProcessor, IdentityDataProcessor
from algorithms import PolicyGradient
from common.utils import TrajectoryResults, SplitResults
from common.tree import BinaryTree
from simulation.trajectory_sampler import TrajectorySampler

import json
import io
from tqdm import tqdm
import copy
from adam.adam import Adam

import os


# Class Implementation
class PolicyGradientSplit(PolicyGradient):
    def __init__(
            self, lr: np.array = None,
            lr_strategy: str = "constant",
            estimator_type: str = "REINFORCE",
            initial_theta: np.array = None,
            ite: int = 100,
            batch_size: int = 1,
            env: BaseEnv = None,
            policy: BasePolicy = None,
            data_processor: BaseProcessor = IdentityDataProcessor(),
            directory: str = "",
            verbose: bool = False,
            natural: bool = False,
            checkpoint_freq: int = 1,
            n_jobs: int = 1,
            split_grid: np.array = None,
            max_splits: int = 10,
            baselines: str = None
    ) -> None:
        # Class' parameter with checks
        err_msg = "[PG_split] lr must be positive!"
        assert lr[0] > 0, err_msg
        self.lr = lr[0]

        err_msg = "[PG_split] lr_strategy not valid!"
        assert lr_strategy in ["constant", "adam"], err_msg
        self.lr_strategy = lr_strategy

        err_msg = "[PG_split] estimator_type not valid!"
        assert estimator_type in ["REINFORCE", "GPOMDP"], err_msg
        self.estimator_type = estimator_type

        err_msg = "[PG_split] initial_theta has not been specified!"
        assert initial_theta is not None, err_msg
        self.thetas = np.array(initial_theta)
        self.dim = len(self.thetas)

        err_msg = "[PG_split] env is None."
        assert env is not None, err_msg
        self.env = env

        err_msg = "[PG_split] policy is None."
        assert policy is not None, err_msg
        self.policy = policy

        err_msg = "[PG_split] data processor is None."
        assert data_processor is not None, err_msg
        self.data_processor = data_processor

        err_msg = "[PG_split] split grid is None."
        assert split_grid is not None, err_msg
        self.split_grid = split_grid

        os.makedirs(directory, exist_ok=True)
        self.directory = directory

        # Other class' parameters
        self.ite = ite
        self.batch_size = batch_size
        self.verbose = verbose
        self.natural = natural
        self.checkpoint_freq = checkpoint_freq
        self.n_jobs = n_jobs
        self.baselines = baselines
        # self.parallel_computation = bool(self.n_jobs != 1)
        self.dim_action = self.env.action_dim
        self.dim_state = self.env.state_dim

        # Useful structures
        self.theta_history = dict.fromkeys([i for i in range(self.ite)], np.array(0))
        self.time = 0
        self.performance_idx = np.zeros(ite, dtype=np.float64)
        self.best_theta = np.zeros(self.dim, dtype=np.float64)
        self.best_performance_theta = -np.inf
        self.sampler = TrajectorySampler(
            env=self.env, pol=self.policy, data_processor=self.data_processor
        )
        self.deterministic_curve = np.zeros(self.ite)

        # init the theta history
        self.theta_history[self.time] = copy.deepcopy(self.thetas)

        # create the adam optimizers
        self.adam_optimizer = None
        if self.lr_strategy == "adam":
            self.adam_optimizer = Adam()

        # self.policy_history = BinaryTree()
        # self.policy_history.insert(self.thetas)

        self.policy.history.insert(self.thetas)
        self.max_splits = max_splits
        self.split_done = False
        self.start_split = False
        self.gradient_history = []

        return

    def learn(self) -> None:
        """Learning function"""
        splits = 0
        for i in tqdm(range(self.ite)):
            res = []

            for j in range(self.batch_size):
                tmp_res = self.sampler.collect_trajectory(params=copy.deepcopy(self.thetas))
                res.append(tmp_res)

            # Update performance
            perf_vector = np.zeros(self.batch_size, dtype=np.float64)
            score_vector = np.zeros((self.batch_size, self.env.horizon, self.dim),
                                    dtype=np.float64)
            reward_vector = np.zeros((self.batch_size, self.env.horizon), dtype=np.float64)
            state_vector = np.zeros((self.batch_size, self.env.horizon, self.dim_state), dtype=np.float64)
            for j in range(self.batch_size):
                perf_vector[j] = res[j][TrajectoryResults.PERF]
                reward_vector[j, :] = res[j][TrajectoryResults.RewList]
                state_vector[j, :, :] = res[j][TrajectoryResults.StateList]
                score_vector[j, :, :] = res[j][TrajectoryResults.ScoreList]

            self.performance_idx[i] = np.mean(perf_vector)

            # Update best rho
            self.update_best_theta(current_perf=self.performance_idx[i])

            # Look for a split
            if splits < self.max_splits and self.start_split:
                self.learn_split(score_vector, state_vector, reward_vector)

            if not self.split_done:
                # Compute the estimated gradient
                if self.estimator_type == "REINFORCE":
                    estimated_gradient = np.mean(
                        perf_vector[:, np.newaxis] * np.sum(score_vector, axis=1), axis=0)
                elif self.estimator_type == "GPOMDP":
                    estimated_gradient = self.update_gpomdp(
                        reward_vector=reward_vector, score_trajectory=score_vector
                    )
                else:
                    err_msg = f"[PG] {self.estimator_type} has not been implemented yet!"
                    raise NotImplementedError(err_msg)

                self.gradient_history.append(estimated_gradient)
                self.update_parameters(estimated_gradient)
            else:
                if self.verbose:
                    self.policy.history.print_tree()

                self.policy.history.to_png(f"policy_tree")
                splits += 1

            # Log
            if self.verbose:
                print("*" * 30)
                print(f"Step: {self.time}")
                print(f"Mean Performance: {self.performance_idx[self.time - 1]}")
                print(f"Estimated gradient: {estimated_gradient}")
                print(f"Parameter (new) values: {self.thetas}")
                print(f"Best performance so far: {self.best_performance_theta}")
                print(f"Best configuration so far: {self.best_theta}")
                print("*" * 30)

            # Checkpoint
            if self.time % self.checkpoint_freq == 0:
                self.save_results()

            # save theta history
            self.theta_history[self.time] = copy.deepcopy(self.thetas)

            # time update
            self.time += 1

            # reduce the exploration factor of the policy
            self.policy.reduce_exploration()

            # check if we reached an optimal configuration
            if splits < self.max_splits:
                self.check_local_optima()

        return

    def split(self, score_vector, state_vector, reward_vector, split_state) -> list:
        traj = []
        closest_leaf = self.policy.history.find_closest_leaf(split_state)
        traj_l, traj_r = 0, 0
        score_left = np.zeros((self.batch_size, self.env.horizon, self.dim), dtype=np.float64)
        score_right = np.zeros((self.batch_size, self.env.horizon, self.dim), dtype=np.float64)

        for i in range(len(state_vector)):
            for j in range(len(state_vector[i])):
                if state_vector[i][j] < split_state and (
                        closest_leaf is None or closest_leaf.val[1] is None or state_vector[i][j] >= closest_leaf.val[1]):
                    score_left[i, j, :] = score_vector[i][j]
                    traj_l += 1
                elif state_vector[i][j] >= split_state and (
                        closest_leaf is None or closest_leaf.val[1] is None or state_vector[i][j] <= closest_leaf.val[1]):
                    score_right[i, j, :] = score_vector[i][j]
                    traj_r += 1

            traj.append([traj_l, traj_r])
            traj_l = 0
            traj_r = 0

        reward_trajectory_left = np.sum(np.cumsum(score_left, axis=1) * reward_vector[...,None], axis=1)
        reward_trajectory_right = np.sum(np.cumsum(score_right, axis=1) * reward_vector[...,None], axis=1)

        estimated_gradient_left = np.mean(reward_trajectory_left)
        estimated_gradient_right = np.mean(reward_trajectory_right)

        estimated_gradient = [estimated_gradient_left, estimated_gradient_right]
        reward_trajectory = [reward_trajectory_left, reward_trajectory_right]

        new_thetas = [self.update_parameters(estimated_gradient_left, local=True, split_state=split_state),
                      self.update_parameters(estimated_gradient_right, local=True, split_state=split_state)]

        for i, elem in enumerate(traj):
            if elem[0] > elem[1]:
                traj_l += 1
            else:
                traj_r += 1

        traj = [traj_l, traj_r]
        return [estimated_gradient, reward_trajectory, new_thetas, traj]

    def learn_split(self, score_vector, state_vector, reward_vector) -> None:
        splits = {}

        for i in range(len(self.split_grid)):
            # print("Iteration split:", i + 1)
            # print("split state:", self.split_grid[i], i)
            res = self.split(score_vector, state_vector, reward_vector, self.split_grid[i])

            reward_trajectory = res[SplitResults.RewardTrajectories]
            estimated_gradient = res[SplitResults.Gradient]
            trajectories = res[SplitResults.ValidTrajectories]
            thetas = res[SplitResults.SplitThetas]

            gradient_norm = np.linalg.norm(estimated_gradient[0]) + np.linalg.norm(estimated_gradient[1])

            # if self.check_split(reward_trajectory[0], reward_trajectory[1], trajectories, estimated_gradient[0], estimated_gradient[1]):
            if self.check_split(reward_trajectory[0], reward_trajectory[1]):
                splits[self.split_grid[i]] = [thetas, True, gradient_norm]

            else:
                splits[self.split_grid[i]] = [thetas, False, gradient_norm]

            # print("norm left: ", np.linalg.norm(grad_tmp[0]))
            # print("norm right: ", np.linalg.norm(grad_tmp[1]))
            # print("sum norm: ", np.linalg.norm(grad_tmp[0])+np.linalg.norm(grad_tmp[1]))

            # print(estimated_gradient[0], estimated_gradient[1])

        valid_splits = {key: value for key, value in splits.items() if value[1] is True}
        print("Valid splits: ", valid_splits)
        if valid_splits:
            split = max(valid_splits.items(), key=lambda x: x[1][2])

            best_split_thetas = split[1][0]
            best_split_state = split[0]

            if self.verbose:
                print("Split result: ", best_split_thetas)
                print("Split state: ", best_split_state)

            self.policy.history.insert(best_split_thetas, best_split_state.item())
            self.split_done = True
            self.thetas = np.array(self.policy.history.get_new_policy())
            self.policy.update_policy_params()
            self.dim = len(self.thetas)

            index = np.argwhere(self.split_grid == best_split_state)
            self.split_grid = np.delete(self.split_grid, index)
        else:
            print("No split found!")

    def update_parameters(self, estimated_gradient, local=False, split_state=None):
        new_theta = None
        old_theta = self.thetas
        # Update parameters
        if split_state is not None:
            old_theta = self.policy.history.find_closest_leaf(split_state).val[0]

        if self.lr_strategy == "constant":
            new_theta = old_theta + self.lr * estimated_gradient
        elif self.lr_strategy == "adam":
            adaptive_lr = self.adam_optimizer.next(estimated_gradient)
            new_theta = old_theta + adaptive_lr
        else:
            err_msg = f"[PG] {self.lr_strategy} not implemented yet!"
            raise NotImplementedError(err_msg)

        if local:
            return new_theta
        else:
            self.thetas = new_theta

    def compute_const(self, left, right):
        res = np.multiply(left, right)
        var = np.var(res)

        # print("Mean", np.mean(res))
        # print("Var", var)
        return np.sqrt(var * 1.96)

    # todo fai moltiplicazione element wise, poi argmin e splitta e rigenera griglia solo su spazio parametro
    # todo con gradienti più negativi
    def check_local_optima(self) -> None:
        if len(self.gradient_history) <= 1:
            self.start_split = False
            return

        # Case where a split just happened so no need to check for local optima
        # Reset gradient history to match the new number of parameters
        if self.split_done:
            self.start_split = False
            self.split_done = False
            self.gradient_history = []
            return

        latest_two = self.gradient_history[-2:]
        # res = np.multiply(latest_two[0], latest_two[1])
        # res = np.dot(latest_two[0], latest_two[1])
        #
        # self.start_split = all(val < 0 for val in res)
        #
        # self.start_split = res < 0
        # if self.start_split:
        #     print("Optimal configuration found!")
        if np.dot(latest_two[0], latest_two[1]) < 0:
            res = np.multiply(latest_two[0], latest_two[1])
            best_region = np.argmin(res)

            # print("AO ANNAMO***************: ", latest_two[0], latest_two[1], latest_two[0].dtype, latest_two[1].dtype)
            if latest_two[0].size == 1:
                self.start_split = True
                print("Optimal configuration found!")
            else:
                self.start_split = True
                print("Optimal configuration found!")
                print("Splitting on param side: ", res[best_region])
                split_point = self.policy.history.get_father(best_region)
                # self.split_grid = np.linspace(split_point.val[1], -split_point.val[1], 10)
        else:
            self.start_split = False
    
    def compute_p(self, left, right):
        p = np.multiply(left, right)
        return p

    def check_split(self, left, right, delta=0.3):
        p = self.compute_p(left, right)
        z = np.var(p)
        sup = np.max(p)
        
        test = np.sqrt((2 * z * np.log(2/delta))/self.batch_size) + (((7 * np.log(1/delta))/(3 * (self.batch_size- 1))) * sup) + np.mean(p)
        term1 = np.sqrt((2 * z * np.log(2/delta))/self.batch_size)
        term2 = (((7 * np.log(1/delta))/(3 * (self.batch_size- 1))) * sup)
        term3 = np.mean(p)

        print("************************", term1, term2, term3)


        return (test < 0)
    
    # todo dividi su tutto n e non
    # def check_split(self, left, right, n, grad_l, grad_r):
    #     # n = 1e-5 if np.min(n) == 0 else np.min(n)
    #     n = self.batch_size
    #     test = np.dot(grad_l, grad_r) + (self.compute_const(left, right) / np.sqrt(n))

    #     # mx = np.zeros(left.shape[0])
    #     # for b in range(left.shape[0]):
    #     #     mx[b] = left[b, :].reshape(1, -1) @ right[b, :].reshape(-1, 1)
    #     #
    #     # test = mx + (self.compute_const(left, right) / np.sqrt(n))
    #     # if right.any() != 0:
    #     #     pass
    #     # print("Dot product", np.dot(left, right))
    #     # print("c/n:", self.compute_const(left, right)/np.sqrt(n))
    #     # print("test:", test, n)
    #     return (test < 0)

    def generate_grid(self, states_vector) -> None:
        """
        Generate a grid of split points based on the states vector.

        Parameters:
        states_vector (np.array): The states vector to generate the grid from.

        Returns:
        np.array: A grid of split points.
        """
        for i in self.batch_size:
            self.split_grid = np.append(self.split_grid, self.sample_states(states_vector[i], 1, 0.5))
        
    
    def sample_states(self, state_vector, num_samples, gamma) -> np.float64:
        """
        Sample states from a state vector using a geometric distribution.

        Parameters:
        state_vector (np.array): The state vector to sample from.
        num_samples (int): The number of samples to draw.
        p (float): The 'success' probability parameter for the geometric distribution.

        Returns:
        np.array: An array of sampled states.
        """
        # Generate indices using a geometric distribution
        indices = np.random.geometric(gamma, num_samples) - 1  # subtract 1 because geometric distribution starts at 1

        # Modulo operation to ensure indices are within the range of the state vector length
        indices = indices % len(state_vector)

        # Return the sampled states
        return state_vector[indices]

    def save_results(self) -> None:
        results = {
            "performance": np.array(self.performance_idx, dtype=float).tolist(),
            "best_theta": np.array(self.best_theta, dtype=float).tolist(),
            "thetas_history": list(value.tolist() for value in self.theta_history.values()),
            "last_theta": np.array(self.thetas, dtype=float).tolist(),
            "best_perf": float(self.best_performance_theta),
            "performance_det": np.array(self.deterministic_curve, dtype=float).tolist()
        }

        # Save the json
        name = self.directory + "/split_results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
        return