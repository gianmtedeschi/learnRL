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
from common.tree import BinaryTree, Node
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
import time





# Class Implementation
class PolicyGradientSplitMultiDimVM(PolicyGradient):
    def __init__(
            self, lr: np.array = None,
            lr_strategy: str = "constant",
            estimator_type: str = "GPOMDP",
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
            max_splits: int = 1000,
            baselines: str = None,
            alpha: float= 0.1

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
        self.thetas= np.array([initial_theta])
        self.dim = len(self.thetas)
        print(self.thetas, self.dim)

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
        

        # Useful structures
        self.theta_history = dict.fromkeys([i for i in range(self.ite)], np.array(0))
        self.time = 0
        self.performance_idx = np.zeros(ite, dtype=np.float64)
        self.best_theta = np.zeros(self.dim, dtype=np.float64)
        self.best_performance_theta = -np.inf
        self.sampler = TrajectorySampler(
            env=self.env, pol=self.policy, data_processor=self.data_processor
        )
        

        # init the theta history
        self.theta_history[self.time] = copy.deepcopy(self.thetas)

        # create the adam optimizers
        self.adam_optimizer = None
        if self.lr_strategy == "adam":
            self.adam_optimizer = Adam()

        self.policy.history.insert_root(self.thetas)
        self.splitting_param = self.policy.history.get_all_leaves()[0]
        self.splitting_coordinate = 0

        self.max_splits = max_splits
        self.split_done = False
        self.start_split = False
        self.delta=0
        self.split_ite=[]
        self.trial=0
        self.alpha= alpha
        # TESTING PURPOSES
        self.split_grid = np.array([[0], [1], [-1],[2],[-2],[4],[-4],[8],[-8],[16],[-16]])

        

        return

    def learn(self) -> None:
        """Learning function"""
        splits = 0
        axis = 0
        gradient_sum, gradient_mean = 0,0
        for i in tqdm(range(self.ite)):
            res = []

            for j in range(self.batch_size):
                tmp_res = self.sampler.collect_trajectory(params=copy.deepcopy(self.thetas),split=True)
                res.append(tmp_res)

            # Update performance
            perf_vector = np.zeros(self.batch_size, dtype=np.float64)
            score_vector = np.zeros((self.batch_size, self.env.horizon, self.dim,self.env.action_dim),
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
                # Choose the split axis
                if axis == self.dim_state:
                    axis = 0
                # Compute the split grid
                self.generate_grid(states_vector=state_vector, axis=axis, num_samples=50)
                print("Split grid: ", self.split_grid, self.split_grid.shape, self.split_grid.dtype)
                
                # Start the split procedure
                self.learn_split(score_vector[:,:,self.splitting_coordinate], state_vector, reward_vector, axis)

                # Update axis for next iteration
                axis += 1

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

                self.delta = self.compute_delta(gradient_mean, gradient_sum, estimated_gradient, i+1)
                gradient_sum += estimated_gradient
                gradient_mean = gradient_sum/(i+1)
                self.update_parameters(estimated_gradient)
                print("Gradient:",estimated_gradient)
            else:
                name= self.directory + "/policy_tree"
                self.policy.history.to_png(name)
                splits +=1
                self.split_ite.append(i)
                gradient_sum = 0
                gradient_mean = 0
                self.delta = 0
                axis = 0
                

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
            else:
                print("Max splits reached!")
                self.split_done=False
        return
    def compute_phi(self,state):
        
        num_leaves= len(self.policy.history.get_all_leaves())
        split_states=[]
        for i in range(num_leaves):
                split_states[i]= self.policy.history.nodes[self.policy.history.get_all_leaves()[i].father_id].val[1][1]
        split_states=np.unique(split_states)
        
        intervals = [(-np.inf, split_states[0])] + [(split_states[i], split_states[i+1]) for i in range(len(split_states)-1)] + [(split_states[-1], np.inf)]
        result_vectors = []
        for interval in intervals:
              vector = np.zeros_like(state)
              if interval[0] < state <= interval[1]:
                     vector[i] = 1
              result_vectors.append(vector)
        return result_vectors



    
        



           

        
        def phi(s):
        zeros = torch.zeros_like(s)
        ones = torch.ones_like(s)
        a = torch.where(s<0.76, ones, zeros)
        b = torch.where(torch.logical_and(s>=0.76, s<1), ones, zeros)
        c = torch.where(s>=1, ones, zeros)
        _phi = torch.stack((a,b,c), -1).squeeze()
        #print(_phi.shape)
        return _phi

    def split(self, score_vector, state_vector, reward_vector, split_state) -> list:
        traj = []

        closest_leaf = self.policy.history.find_region_leaf(split_state)
        lower_vertex = self.policy.history.get_lower_vertex(closest_leaf, self.dim_state)
        upper_vertex = self.policy.history.get_upper_vertex(closest_leaf,  self.dim_state)

        left_upper = np.zeros(self.dim_state)
        left_lower = np.zeros(self.dim_state)
        right_upper = np.zeros(self.dim_state)
        right_lower = np.zeros(self.dim_state)

        left_lower = lower_vertex
        for i in range(self.dim_state):
            left_upper[i] = upper_vertex[i]
            if i == split_state[0]:
                left_upper[i] = split_state[1]
        
        right_upper = upper_vertex
        for i in range(self.dim_state):
            right_lower[i] = lower_vertex[i]
            if i == split_state[0]:
                right_lower[i] = split_state[1]
        
        
        # print("Destra:", right_lower, right_upper)
        # print("Sinistra:", left_lower, left_upper)


        traj_l, traj_r = 0, 0
        # score_left = np.zeros((self.batch_size, self.env.horizon, self.dim), dtype=np.float64)
        # score_right = np.zeros((self.batch_size, self.env.horizon, self.dim), dtype=np.float64)


        score_left = np.zeros((self.batch_size, self.env.horizon, self.env.action_dim), dtype=np.float64)
        score_right = np.zeros((self.batch_size, self.env.horizon, self.env.action_dim), dtype=np.float64)

        # for i in range(len(state_vector)):
        #     for j in range(len(state_vector[i])):
        #         # TODO cambia if basandola su vertici regione
        #         if state_vector[i][j].all() < split_state.all() and (
        #                 closest_leaf is None or closest_leaf.val[1] is None or state_vector[i][j] >= closest_leaf.val[1]):
        #             score_left[i, j, :] = score_vector[i][j]
        #             traj_l += 1
        #         elif state_vector[i][j].all() >= split_state.all() and (
        #                 closest_leaf is None or closest_leaf.val[1] is None or state_vector[i][j] <= closest_leaf.val[1]):
        #             score_right[i, j, :] = score_vector[i][j]
        #             traj_r += 1
        
        for i in range(len(state_vector)):
            for j in range(len(state_vector[i])):
                if np.all(state_vector[i][j] > left_lower) and np.all(state_vector[i][j] <= left_upper):
                    score_left[i, j, :] = score_vector[i][j]
                    traj_l += 1
                elif np.all(state_vector[i][j] > right_lower) and np.all(state_vector[i][j] <= right_upper):
                    score_right[i, j, :] = score_vector[i][j]
                    traj_r += 1

            traj.append([traj_l, traj_r])
            traj_l = 0
            traj_r = 0


        reward_trajectory_left = np.sum(np.cumsum(score_left, axis=1) * reward_vector[...,None], axis=1)
        reward_trajectory_right = np.sum(np.cumsum(score_right, axis=1) * reward_vector[...,None], axis=1)

        # print("Reward trajectory left: ", reward_trajectory_left)
        # print("Reward trajectory right: ", reward_trajectory_right)

        estimated_gradient_left = np.mean(reward_trajectory_left,axis=0)
        estimated_gradient_right = np.mean(reward_trajectory_right,axis=0)

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

    def learn_split(self, score_vector, state_vector, reward_vector, axis) -> None:
        splits = {}

        for i in range(len(self.split_grid)):
            # print("Iteration split:", i + 1)
            # print("split state:", self.split_grid[i], i)
            res = self.split(score_vector, state_vector, reward_vector, np.array([axis, self.split_grid[i][axis]]))

            reward_trajectory = res[SplitResults.RewardTrajectories]
            estimated_gradient = res[SplitResults.Gradient]
            trajectories = res[SplitResults.ValidTrajectories]
            thetas = res[SplitResults.SplitThetas]

            gradient_norm = np.linalg.norm(estimated_gradient[0]) + np.linalg.norm(estimated_gradient[1])
            # print("Point: ", self.split_grid[i])
            # print("Gradient mean:", estimated_gradient[0], estimated_gradient[1])
            # print("Thetas: ", thetas)
            # if self.check_split(reward_trajectory[0], reward_trajectory[1], trajectories, estimated_gradient[0], estimated_gradient[1]):
            
            key = tuple([axis, self.split_grid[i][axis]])  
            
            if self.check_split(reward_trajectory[0], reward_trajectory[1],self.alpha):
                splits[key] = [thetas, True, gradient_norm]

            else:
                splits[key] = [thetas, False, gradient_norm]

            # print("norm left: ", np.linalg.norm(grad_tmp[0]))
            # print("norm right: ", np.linalg.norm(grad_tmp[1]))
            # print("sum norm: ", np.linalg.norm(grad_tmp[0])+np.linalg.norm(grad_tmp[1]))

            # print(estimated_gradient[0], estimated_gradient[1])

        valid_splits = {key: value for key, value in splits.items() if value[1] is True}
        valid_splits={key:(value[0],self.policy.history.check_already_existing_split(key),value[2]) for key,value in valid_splits.items()}
        valid_splits= {key: value for key,value in valid_splits.items() if value[1] is True}

        print("Valid splits: ", valid_splits)
        if valid_splits:
            split = max(valid_splits.items(), key=lambda x: x[1][2])

            best_split_thetas = split[1][0]
            best_split_state = split[0]

            #if self.verbose:
            print("Split result: ", best_split_thetas)
            print("Split state: ", best_split_state)

            # update tree policy
            # self.policy.history.insert(best_split_thetas, self.father_id, best_split_state.item())
            self.policy.history.insert(np.array(best_split_thetas), self.father_id, best_split_state)

            self.split_done = True
            self.thetas = np.array(self.policy.history.get_current_policy())
            print("New thetas: ", self.thetas)
            self.dim = len(self.thetas)
            # adam update
            if self.lr_strategy == "adam":
                index_of_split = list(splits.keys()).index(best_split_state)
                self.adam_optimizer.update_params(local=False, coord=self.splitting_coordinate, index=index_of_split)
                

            # index = np.argwhere(self.split_grid == best_split_state)
            # self.split_grid = np.delete(self.split_grid, index)
        else:
            print("No split found!")
            self.split_done=False

    def update_parameters(self, estimated_gradient, local=False, split_state=None):
        new_theta = None
        coord=None
        old_theta = self.thetas
        # Update parameters
        if split_state is not None:
            old_theta = self.policy.history.find_region_leaf(split_state).val[0]
            coord = self.splitting_coordinate

        if self.lr_strategy == "constant":
            new_theta = old_theta + self.lr * estimated_gradient

        elif self.lr_strategy == "adam":
            adaptive_lr = self.adam_optimizer.next(estimated_gradient,coord=coord,local=local)
            new_theta = old_theta + adaptive_lr
        else:
            err_msg = f"[PG] {self.lr_strategy} not implemented yet!"
            raise NotImplementedError(err_msg)

        if local:
            return new_theta
        else:
            self.thetas = new_theta
            self.policy.history.update_all_leaves(self.thetas)
            self.policy.history.to_png(self.directory + "/policy_tree")

    def compute_const(self, left, right):
        res = np.multiply(left, right)
        var = np.var(res)

        return np.sqrt(var * 1.96)

    

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
    
    def uniformity_test(self, n, R):
        return (2 * n * R**2 > stats.chi2.ppf(0.995, 2))



    def check_split(self, left, right, alpha):

        test = False
        angle = self.compute_angle(left, right)

        #print("DEBUG:", left, right, angle)
        N = len(angle)
        C_1 = np.sum(np.cos(angle))
        S_1 = np.sum(np.sin(angle))
        R_1 = np.sqrt(C_1**2 + S_1**2)
        
        C = C_1/N
        S = S_1/N
        R = R_1/N
        
        T_1 = np.arctan2(S,C)

        if T_1 < 0:
            T_1 += np.pi
        
        print("N: ", N)
        if not self.uniformity_test(N, R):
            print("Uniformity test failed")
            return False
        elif R <= 2/3:
            conf_interval = [T_1 - np.arccos(np.sqrt(2*N*(2*R_1**2 - N*stats.chi2.ppf(1 - alpha, 1))/((R_1**2)*(4*N-stats.chi2.ppf(1 - alpha, 1))))),
                             T_1 + np.arccos(np.sqrt(2*N*(2*R_1**2 - N*stats.chi2.ppf(1 - alpha, 1))/((R_1**2)*(4*N-stats.chi2.ppf(1 - alpha, 1)))))]
        else:
            conf_interval = [T_1 - np.arccos(np.sqrt(N**2 - (N**2 - R_1**2)*np.exp((stats.chi2.ppf(1 - alpha, 1))/N))/R_1),
                             T_1 + np.arccos(np.sqrt(N**2 - (N**2 - R_1**2)*np.exp((stats.chi2.ppf(1 - alpha, 1))/N))/R_1)]


        

        
        print("[", conf_interval[0],",",conf_interval[1],"]")
        
        

        if conf_interval[0] > np.pi/2:
            test = True
        else:
            test = False
                
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
    
    def generate_grid(self, states_vector, axis, num_samples=1) -> np.array:
        """
        Generate a grid of split points based on the occupancy of sampled trajectories.

        Parameters:
        states_vector (np.array): The matrix of sampled trajectories to generate the grid from.

        axis (int): The axis to generate the grid on.

        Returns:
        np.array: A grid of split points.
        """

        # Get the valid region for the current splitting parameter
        valid_region = self.policy.history.get_region(self.splitting_param, self.dim_state)
        print("Valid region: ", valid_region)

        # Draw samples from a geometric distribution
        samples = np.random.geometric(1 - self.env.gamma, num_samples)
        samples = np.clip(samples, 0, self.env.horizon - 1)

        # Get the points to sample from the trajectories
        points = np.linspace(0, num_samples - 1, num_samples, dtype=int) % self.batch_size

        # Get the points to sample from the trajectories
        tmp_grid = states_vector[points, samples]

        # Generate a mask to filter only state in the valid region
        mask = np.zeros((num_samples, self.dim_state), dtype=bool)
        for j in range(num_samples):
            for i in range(self.dim_state):
                mask[j][i] = (tmp_grid[j][i] >= valid_region[i][0]) & (tmp_grid[j][i] <= valid_region[i][1])
        
        # Generate the grid based on the valid region
        tmp_grid = tmp_grid * mask
        

        # Convert each unique tuple back to an array and set a value only in the position defined by axis
        self.split_grid = np.unique(np.array([self.set_value_at_axis(np.array(x).ravel(), axis) for x in tmp_grid]), axis=0)

    
    def set_value_at_axis(self, arr, axis):
        # Create a new array with the same shape as arr and set all elements to zero
        new_arr = np.zeros(arr.shape)

        # Set the element at the position defined by axis to the corresponding value in arr
        new_arr[axis] = arr[axis]

        return new_arr

    def check_local_optima(self, not_avg_gradient, n=10) -> None:
        # Case where a split just happened so no need to check for local optima
        # Reset gradient history to match the new number of parameters
        if self.split_done:
            self.start_split = False
            self.split_done = False
            self.splitting_coordinate = None
            self.trial = 0
            self.gradient_sum = 0
            return

        
        # mean = np.mean(self.gradient_history[-n:], axis=0)
        # self.mean = self.gradient_sum/self.ite
        
        delta = np.linalg.norm(self.delta)
        self.delta = 0

        print("Delta gradient mean: ", delta)

        if np.isclose(delta, 0, atol=1):
            # print(not_avg_gradient.shape)
            var = np.var(not_avg_gradient, axis=0)
            best_region = np.argmax(np.sum(var, axis=1))
            print("Variance: ", var)

            # first iteration case
            if len(var) == 1:
                self.start_split = True
                
                self.father_id = 0
                self.splitting_param = self.policy.history.get_all_leaves()[0]
                self.splitting_coordinate = 0
                print("Optimal configuration found!")
            
            # multidimensional case
            else:
                if best_region == self.splitting_coordinate and self.trial != 0:
                    print("Same region, changing trial")
                    best_region = np.argsort(np.sum(var, axis=1))[::-1][(self.splitting_coordinate + self.trial) % len(var)]
                
                self.start_split = True
                print("Optimal configuration found!")
                print("Splitting on param side: ", self.policy.history.get_all_leaves()[best_region].val[0])

                # usefull structures
                self.father_id = self.policy.history.get_all_leaves()[best_region].node_id
                self.splitting_param = self.policy.history.get_all_leaves()[best_region]
                self.splitting_coordinate = best_region
                self.trial += 1

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
            "split_ite":self.split_ite
            # "performance_det": np.array(self.deterministic_curve, dtype=float).tolist()
        }

        # Save the json
        name = self.directory + "/split_VM_md_results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
        return
    

    def update_gpomdp(
            self, reward_vector: np.array,
            score_trajectory: np.array
    ) -> np.array:
        gamma = self.env.gamma
        horizon = self.env.horizon
        gamma_seq = (gamma * np.ones(horizon, dtype=np.float64)) ** (np.arange(horizon))
        rolling_scores = np.cumsum(score_trajectory, axis=1) + 1e-10 #NxHxPxd_a
        p, d_a = rolling_scores.shape[2], rolling_scores.shape[3]
        # rolling_scores = rolling_scores.reshape(rolling_scores.shape[0], rolling_scores.shape[1], -1)


        if self.baselines == "avg":
            b = np.mean(reward_vector[...,None], axis=0)
        elif self.baselines == "peters":
            b = np.sum(rolling_scores ** 2 * reward_vector[...,None][...,None], axis=0) / np.sum(rolling_scores ** 2, axis=0)
        else:
            b = np.zeros(1)

        reward_trajectory = (reward_vector[...,None][...,None] - b[np.newaxis,...]) * rolling_scores
        # reward_trajectory = reward_trajectory.reshape(reward_trajectory.shape[0], reward_trajectory.shape[1], p, d_a)

        not_avg_gradient = np.sum(gamma_seq[...,None][...,None] * reward_trajectory, axis=1)

        estimated_gradient = np.mean(
            np.sum(gamma_seq[...,None][...,None] * reward_trajectory, axis=1),
            axis=0)

        return estimated_gradient, not_avg_gradient
    
    def compute_delta(self, mean, summation, gradient, n):
        new_mean = (summation + gradient)/n
        return new_mean - mean