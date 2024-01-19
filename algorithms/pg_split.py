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
"""



# imports
import numpy as np
from envs.base_env import BaseEnv
from policies import BasePolicy, GaussianPolicy
from data_processors import BaseProcessor, IdentityDataProcessor
from algorithms import PolicyGradient

# todo
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
class PG_Split(PolicyGradient):
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
            split_grid: np.array = None
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
        # self.parallel_computation = bool(self.n_jobs != 1)
        self.dim_action = self.env.da
        self.dim_state = self.env.ds

        # Useful structures
        self.theta_history = np.zeros((self.ite, self.dim), dtype=np.float64)
        self.time = 0
        self.performance_idx = np.zeros(ite, dtype=np.float64)
        self.best_theta = np.zeros(self.dim, dtype=np.float64)
        self.best_performance_theta = -np.inf
        self.sampler = TrajectorySampler(
            env=self.env, pol=self.policy, data_processor=self.data_processor
        )
        self.deterministic_curve = np.zeros(self.ite)

        # init the theta history
        self.theta_history[self.time, :] = copy.deepcopy(self.thetas)

        # create the adam optimizers
        self.adam_optimizer = None
        if self.lr_strategy == "adam":
            self.adam_optimizer = Adam()
        return

    def learn(self) -> None:
        """Learning function"""
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
            state_vector = np.zeros((self.batch_size, self.env.horizon), dtype=np.float64)
            for j in range(self.batch_size):
                perf_vector[j] = res[j][TrajectoryResults.PERF]
                reward_vector[j, :] = res[j][TrajectoryResults.RewList]
                score_vector[j, :, :] = res[j][TrajectoryResults.ScoreList]
                state_vector[j, :, :, :] = res[j][TrajectoryResults.StateList]

            self.performance_idx[i] = np.mean(perf_vector)

            # Update best rho
            self.update_best_theta(current_perf=self.performance_idx[i])

            # Look for a split
            self.learn_split(score_vector, state_vector, reward_vector)

            #todo se lo split è avvenuto non devo stimare il gradiente ma ricominciare con la nuova policy

            # Compute the estimated gradient
            if self.estimator_type == "REINFORCE":
                estimated_gradient = np.mean(
                    perf_vector[:, np.newaxis] * np.sum(score_vector, axis=1), axis=0)
            elif self.estimator_type == "GPOMDP":
                estimated_gradient = self.update_gpomdp(
                    reward_trajectory=reward_vector, score_trajectory=score_vector
                )
                print("Estimated Gradient:", estimated_gradient)
            else:
                err_msg = f"[PG] {self.estimator_type} has not been implemented yet!"
                raise NotImplementedError(err_msg)

            self.update_parameters(estimated_gradient)

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
            self.theta_history[self.time, :] = copy.deepcopy(self.thetas)

            # time update
            self.time += 1

            # reduce the exploration factor of the policy
            self.policy.reduce_exploration()
        return

    #todo adjust split condition
    def split(self, score_vector, state_vector, reward_vector, split_state) -> list:
        tmp_r, tmp_l = [], []
        score_left, score_right = [], []
        traj = []

        for i in range(len(state_vector)):
            for j in range(len(state_vector[i])):
                if state_vector[i][j] < split_state and (
                        closest_leaf is None or closest_leaf.val[1] is None or states[i][j] >= closest_leaf.val[1]):
                    tmp_l.append(score_vector[i][j])
                    tmp_r.append(0)
                    traj_l += 1
                elif state_vector[i][j] >= split_state and (
                        closest_leaf is None or closest_leaf.val[1] is None or states[i][j] <= closest_leaf.val[1]):
                    tmp_l.append(0)
                    tmp_r.append(score_vector[i][j])
                    traj_r += 1
                else:
                    tmp_r.append(0)
                    tmp_l.append(0)

            score_left.append(tmp_l)
            score_right.append(tmp_r)
            traj.append([traj_l, traj_r])
            traj_l = 0
            traj_r = 0
            tmp_r, tmp_l = [], []

        reward_trajectory_left = np.cumsum(score_left, axis=1) * reward_vector
        reward_trajectory_right = np.cumsum(score_right, axis=1) * reward_vector
        estimated_gradient_left = np.mean(reward_trajectory_left, axis=1)
        estimated_gradient_right = np.mean(reward_trajectory_right, axis=1)

        estimated_gradient = [estimated_gradient_left, estimated_gradient_right]
        reward_trajectory = [reward_trajectory_left, reward_trajectory_right]

        #todo here you are setting self.thetas!!!!!!
        new_thetas = [self.update_parameters(estimated_gradient_left), self.update_parameters(estimated_gradient_right)]

        for i, elem in enumerate(traj):
            if elem[0] > elem[1]:
                traj_l += 1
            else:
                traj_r += 1

        traj = [traj_l, traj_r]

        return [estimated_gradient, reward_trajectory, new_thetas, traj]

    def learn_split(self, score_vector, state_vector, reward_vector) -> None:
        gradient_norms, correct_splits, splits = [], [], []
        correct_mask = []
        test = []
        tmp = []

        for i in range(len(self.split_grid)):
            # print("Iteration split:", i + 1)
            # print("split state:", self.split_grid[i], i)
            res = self.split(score_vector, state_vector, reward_vector, self.split_grid[i])

            reward_trajectory = res[SplitResults.RewardTrajectories]
            estimated_gradient = res[SplitResults.Gradient]
            trajectories = res[SplitResults.ValidTrajectories]
            thetas = res[SplitResults.SplitThetas]

            if self.check_split(reward_trajectory[0], reward_trajectory[1], trajectories):
                correct_mask.append(True)
            else:
                correct_mask.append(False)

            splits.append(thetas)

            # print("norm left: ", np.linalg.norm(grad_tmp[0]))
            # print("norm right: ", np.linalg.norm(grad_tmp[1]))
            # print("sum norm: ", np.linalg.norm(grad_tmp[0])+np.linalg.norm(grad_tmp[1]))

            gradient_norms.append(np.linalg.norm(estimated_gradient[0]) + np.linalg.norm(estimated_gradient[1]))

        for i in range(len(correct_mask)):
            if correct_mask[i]:
                tmp.append(gradient_norms[i])
                test.append(splits[i])
                correct_splits.append(self.split_grid[i])
                print(gradient_norms[i], i + 1)

        if correct_splits.size == 0:
            print("No split found!")
        else:
            max_norm = np.argmax(tmp)
            best_split_thetas = test[max_norm]
            best_split_state = correct_splits[max_norm]

            print("Split result: ", best_split_thetas, tmp[max_norm], correct_splits[max_norm])
            print("Split state :", best_split_state)

            self.policy = GaussianPolicy(
                parameters=best_split_thetas,
                dim_state=self.env.ds,
                dim_action=self.env.da,
                std_dev=0.1,
                std_decay=0,
                std_min=1e-6,
                multi_linear=False,
                constant=True
            )

    def update_parameters(self, estimated_gradient):
        # Update parameters
        if self.lr_strategy == "constant":
            self.thetas = self.thetas + self.lr * estimated_gradient
        elif self.lr_strategy == "adam":
            adaptive_lr = self.adam_optimizer.next(estimated_gradient)
            self.thetas = self.thetas + adaptive_lr
        else:
            err_msg = f"[PG] {self.lr_strategy} not implemented yet!"
            raise NotImplementedError(err_msg)

    def compute_const(self, left, right):
        res = []
        # print(left.shape, right.shape)
        for i in range(len(left)):
            res.append(np.dot(left[i], right[i]))
            # print(left[i], right[i])

        # print("Left", np.var(left), np.mean(left))
        # print("Right", np.var(right), np.mean(right))
        var = np.var(res)

        # print("Mean", np.mean(res))
        # print("Var", np.mean(var))
        return np.sqrt(var * 1.96)

    def check_split(self, left, right, n):
        n = 1e-5 if np.min(n) == 0 else np.min(n)
        test = np.dot(left, right) + (self.compute_const(left, right) / np.sqrt(n))
        # print("Dot product", np.dot(left, right))
        # print("c/n:", self.compute_const(left, right)/np.sqrt(n))
        # print("test:", test, n)

        return test < 0



