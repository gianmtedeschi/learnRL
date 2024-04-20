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

"""
criterio convergenza: norma della media degli ultimi n gradienti è vicina a 0

per scegliere regione guardo componente del gradiente con varianza più grande ---> varianza sulle traiettorie ---> prendo gradienti non mediati e faccio varianze colonne gradienti 

sub sampling sui punti di split ?????

se nessun punto e valido guardi seconda regione con varianza più grande

nessun punto in nessuna regione ---> convergenza

Crea struttura dati con varianza delle coordinate gradiente per mantenerti dove andare a cercare i punti 
Crea una funzione find interval che ti restituisce gli estremi dell'intervallo di una regione --> guarda gli ultimi due padri
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
import scipy
import scipy.stats as stats
import math 
import astropy.stats.circstats as circ

import json
import io
from tqdm import tqdm
import copy
from adam.adam import Adam

import os


# Class Implementation
class PolicyGradientSplitVM(PolicyGradient):
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
            max_splits: int = 100,
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

        # err_msg = "[PG_split] split grid is None."
        # assert split_grid is not None, err_msg
        # self.split_grid = split_grid

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
        print("Dim state: ", self.dim_state)

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

        self.policy.history.insert_root(self.thetas)
        self.splitting_param = self.policy.history.get_all_leaves()[0]

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
                # Compute the split grid
                self.generate_grid(states_vector=state_vector, num_samples=20)
                print("Split grid: ", self.split_grid, self.split_grid.shape, self.split_grid.dtype)
                self.learn_split(score_vector[:,:,self.splitting_coordinate], state_vector, reward_vector)

            if not self.split_done:
                # Compute the estimated gradient
                if self.estimator_type == "REINFORCE":
                    estimated_gradient = np.mean(
                        perf_vector[:, np.newaxis] * np.sum(score_vector, axis=1), axis=0)
                elif self.estimator_type == "GPOMDP":
                    estimated_gradient, not_avg_gradient = self.update_gpomdp(
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
                self.check_local_optima(not_avg_gradient)
        return

    def split(self, score_vector, state_vector, reward_vector, split_state) -> list:
        traj = []

        closest_leaf = self.policy.history.find_region_leaf(split_state)
        valid_region = self.policy.history.get_region(self.splitting_param)

        traj_l, traj_r = 0, 0
        # score_left = np.zeros((self.batch_size, self.env.horizon, self.dim), dtype=np.float64)
        # score_right = np.zeros((self.batch_size, self.env.horizon, self.dim), dtype=np.float64)


        score_left = np.zeros((self.batch_size, self.env.horizon, 1), dtype=np.float64)
        score_right = np.zeros((self.batch_size, self.env.horizon, 1), dtype=np.float64)

        mask = (state_vector >= valid_region[0]) & (state_vector <= valid_region[1])
        filtered_state_vector = state_vector[mask]

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
            print("Point: ", self.split_grid[i])
            print("Gradient mean:", estimated_gradient[0], estimated_gradient[1])
            print("Thetas: ", thetas)
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

            # update tree policy
            self.policy.history.insert(best_split_thetas, self.father_id, best_split_state.item())

            self.split_done = True
            self.thetas = np.array(self.policy.history.get_current_policy()).ravel()
            print("New thetas: ", self.thetas)
            self.policy.update_policy_params()
            self.dim = len(self.thetas)

            # index = np.argwhere(self.split_grid == best_split_state)
            # self.split_grid = np.delete(self.split_grid, index)
        else:
            print("No split found!")

    def update_parameters(self, estimated_gradient, local=False, split_state=None):
        new_theta = None
        old_theta = self.thetas
        # Update parameters
        if split_state is not None:
            old_theta = self.policy.history.find_region_leaf(split_state).val[0]

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

        return np.sqrt(var * 1.96)

    # todo fai moltiplicazione element wise, poi argmin e splitta e rigenera griglia solo su spazio parametro
    # todo con gradienti più negativi
    # def check_local_optima(self) -> None:
    #     if len(self.gradient_history) <= 1:
    #         self.start_split = False
    #         return

    #     # Case where a split just happened so no need to check for local optima
    #     # Reset gradient history to match the new number of parameters
    #     if self.split_done:
    #         self.start_split = False
    #         self.split_done = False
    #         self.gradient_history = []
    #         return

    #     latest_two = self.gradient_history[-2:]
    #     # res = np.multiply(latest_two[0], latest_two[1])
    #     # res = np.dot(latest_two[0], latest_two[1])
    #     #
    #     # self.start_split = all(val < 0 for val in res)
    #     #
    #     # self.start_split = res < 0
    #     # if self.start_split:
    #     #     print("Optimal configuration found!")
    #     if np.dot(latest_two[0], latest_two[1]) < 0:
    #         res = np.multiply(latest_two[0], latest_two[1])
    #         best_region = np.argmin(res)

    #         # print("AO ANNAMO***************: ", latest_two[0], latest_two[1], latest_two[0].dtype, latest_two[1].dtype)
    #         if latest_two[0].size == 1:
    #             self.start_split = True
    #             print("Optimal configuration found!")
    #         else:
    #             self.start_split = True
    #             print("Optimal configuration found!")
    #             print("Splitting on param side: ", res[best_region])
    #             split_point = self.policy.history.get_father(best_region)
    #             # self.split_grid = np.linspace(split_point.val[1], -split_point.val[1], 10)
    #     else:
    #         self.start_split = False
    
    # def compute_p(self, left, right):
    #     p = np.multiply(left, right)
    #     return p

    # def check_split(self, left, right, delta=0.3):
    #     p = self.compute_p(left, right)
    #     z = np.var(p)
    #     sup = np.max(p)
        
    #     test = np.sqrt((2 * z * np.log(2/delta))/self.batch_size) + (((7 * np.log(1/delta))/(3 * (self.batch_size- 1))) * sup) + - np.mean(p)
    #     term1 = np.sqrt((2 * z * np.log(2/delta))/self.batch_size)
    #     term2 = (((7 * np.log(1/delta))/(3 * (self.batch_size- 1))) * sup)
    #     term3 = np.mean(p)

    #     print("************************", term1, term2, term3)

    #     return (test < 0)

    def compute_p(self, left, right):
        p = np.multiply(left, right)
        return p
    
    def compute_angle(self,left,right):
        dot_products= np.sum(left*right,axis=1)
        left_norms = np.linalg.norm(left,axis=1)
        right_norms = np.linalg.norm(right,axis=1)

        non_zero_indices = np.logical_and(left_norms != 0, right_norms != 0)
    
    
        cos_angles = np.zeros_like(dot_products)
        cos_angles[non_zero_indices] = dot_products[non_zero_indices] / (left_norms[non_zero_indices] * right_norms[non_zero_indices])
      
        cos_angles = np.clip(cos_angles,-1.0,1.0)
        angles = np.arccos(cos_angles)
        return angles
    
    def check_Von_Mises(self,angles):
        params= circ.vonmisesmle(angles)
        pvalue= stats.kstest(angles,"vonmises",args=(params[0],params[1]))[1]
        if pvalue< 0.1:
            return True
        else:
            return False



    def check_split(self, left, right, delta=0.1):
        test= False
        angle = self.compute_angle(left, right)
        N= len(angle)
        C_1= np.sum(np.cos(angle))
        S_1= np.sum(np.sin(angle))
        R_1=np.sqrt(C_1**2 + S_1**2)
        C= C_1/N
        S= S_1/N
        R= R_1/N
        if C<0:
            T_1= np.arctan(S/C) + np.pi
        if C>0 and S<0:
            T_1= np.arctan(S/C) +2*np.pi
        if C>0 and S>=0:
            T_1= np.arctan(S/C)

        if(self.check_Von_Mises(angle)):
          if R<= 2/3:
              conf_interval = [np.degrees(T_1) - np.degrees(np.arccos(np.sqrt(2*N*(2*R_1**2 - N*stats.norm.ppf(1-delta/2)**2)/((R_1**2)*(4*N-stats.norm.ppf(1-delta/2)**2))))),
                               np.degrees(T_1) + np.degrees(np.arccos(np.sqrt(2*N*(2*R_1**2 - N*stats.norm.ppf(1-delta/2)**2)/((R_1**2)*(4*N-stats.norm.ppf(1-delta/2)**2)))))]
          else:
              conf_interval= [np.degrees(T_1) - np.degrees(np.arccos(np.sqrt(N**2 - (N**2 - R_1**2)*np.exp(((stats.norm.ppf(1-delta/2))**2)/N))/R_1)),
                               np.degrees(T_1) + np.degrees(np.arccos(np.sqrt(N**2 - (N**2 - R_1**2)*np.exp(((stats.norm.ppf(1-delta/2))**2)/N))/R_1))]

        else:
           H= (np.cos(2*T_1)*np.sum(np.cos(2*angle)) + np.sin(2*T_1)*np.sum(np.sin(2*angle)))/N
           sigma_hat = np.sqrt(N*(1-H)/(4*R_1**2))
           conf_interval= [np.degrees(T_1) - np.degrees(np.arcsin(sigma_hat*stats.norm.ppf(1-delta/2))), np.degrees(T_1) + np.degrees(np.arcsin(sigma_hat*stats.norm.ppf(1-delta/2)))]

        
        print("[", conf_interval[0],",",conf_interval[1],"]")
        if conf_interval[1]< np.degrees(np.pi/2):
            test=False
        if conf_interval[0]> np.degrees(np.pi/2):
            test= True
        
        
        



        return test

    


    
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
    
    def generate_grid(self, states_vector, num_samples=1) -> np.array:
        """
        Generate a grid of split points based on the occupancy of sampled trajectories.

        Parameters:
        states_vector (np.array): The matrix of sampled trajectories to generate the grid from.

        Returns:
        np.array: A grid of split points.
        """
        valid_region = self.policy.history.get_region(self.splitting_param)
        print("Valid region: ", valid_region)

        samples = np.random.geometric(1 - self.env.gamma, num_samples)
        samples = np.clip(samples, 0, self.env.horizon - 1)

        points = np.linspace(0, num_samples - 1, num_samples, dtype=int) % self.batch_size

        tmp_grid = states_vector[points, samples].ravel()

        mask = (tmp_grid >= valid_region[0]) & (tmp_grid <= valid_region[1])
        self.split_grid = tmp_grid[mask]

    def check_local_optima(self, not_avg_gradient, n=20) -> None:
        if len(self.gradient_history) <= n:
            self.start_split = False
            return

        # Case where a split just happened so no need to check for local optima
        # Reset gradient history to match the new number of parameters
        if self.split_done:
            self.start_split = False
            self.split_done = False
            self.gradient_history = []
            return

        mean = np.mean(self.gradient_history[-n:])
        mean = np.linalg.norm(mean)
        print("Gradient mean: ", mean)

        if np.isclose(mean, 0, atol=0.5):
            print(not_avg_gradient.shape)
            var = np.var(not_avg_gradient, axis=0)
            best_region = np.argmax(var)
            print("Variance: ", var)

            # scalar case
            if var.size == 1:
                self.start_split = True
                
                self.father_id = 0
                self.splitting_param = self.policy.history.get_all_leaves()[0]
                self.splitting_coordinate = 0
                print("Optimal configuration found!")
            
            # multidimensional case
            else:
                self.start_split = True
                print("Optimal configuration found!")
                print("Splitting on param side: ", self.policy.history.get_all_leaves()[best_region].val[0])
                
                # save father id for future insert
                # usefull structures
                self.father_id = self.policy.history.get_all_leaves()[best_region].node_id

                print("Father id: ", self.father_id, self.policy.history.get_all_leaves()[best_region].id_father)
                self.policy.history.to_list(self.policy.history.nodes[self.father_id])
                
                self.splitting_param = self.policy.history.get_all_leaves()[best_region]
                self.splitting_coordinate = best_region

            self.start_split = True
        else:
            self.start_split = False

    def save_results(self) -> None:
        results = {
            "performance": np.array(self.performance_idx, dtype=float).tolist(),
            "best_theta": np.array(self.best_theta, dtype=float).tolist(),
            "thetas_history": list(value.tolist() for value in self.theta_history.values()),
            "last_theta": np.array(self.thetas, dtype=float).tolist(),
            "best_perf": float(self.best_performance_theta),
            # "performance_det": np.array(self.deterministic_curve, dtype=float).tolist()
        }

        # Save the json
        name = self.directory + "/split_VM_results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
        return
    







