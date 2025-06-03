"""Policy Gradient Implementation"""
# todo baseline

# imports
import numpy as np
from envs.base_env import BaseEnv
from policies import BasePolicy
from policies import GaussianPolicy
from data_processors import BaseProcessor, IdentityDataProcessor

# todo
from common.utils import TrajectoryResults

from joblib import Parallel, delayed
from simulation.trajectory_sampler import TrajectorySampler, pg_sampling_worker_bpo, pg_sampling_worker

import json
import io
from tqdm import tqdm
import copy
from adam.adam import Adam
import torch
import torch.optim as optim

from scipy.special import logsumexp

# todo
# maybe in utils?
import os


# Class Implementation
class PolicyGradientBpoMIS:
    """This Class implements Policy Gradient Algorithms via REINFORCE or GPOMDP."""
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
            baselines: str = None,
            checkpoint_freq: int = 1,
            n_jobs: int = 1,
            debug: bool = False,
            seed = 0,
            defensive_batch_size = 0,
            evaluation_batch_size = 100
            # numero behavioural policies, o lista batches tipo [100,100,100] per trovare behavioural
            #se riusare le behavioural policies vecchie, e fino a che numero k ( default 1 )
            # defensive batch
            
    ) -> None:
        # Class' parameter with checks
        # err_msg = "[PG] lr must be positive!"
        # assert lr[0] > 0, err_msg
        self.lr = lr[0]

        err_msg = "[PG] lr_strategy not valid!"
        assert lr_strategy in ["constant", "adam"], err_msg
        self.lr_strategy = lr_strategy

        err_msg = "[PG] estimator_type not valid!"
        assert estimator_type in ["REINFORCE", "GPOMDP"], err_msg
        self.estimator_type = estimator_type

        err_msg = "[PG] initial_theta has not been specified!"
        assert initial_theta is not None, err_msg
        self.thetas = np.array(initial_theta)
        self.thetas_behavioural = np.array(initial_theta)
        self.dim = len(self.thetas)

        err_msg = "[PG] env is None."
        assert env is not None, err_msg
        self.env = env

        err_msg = "[PG] policy is None."
        assert policy is not None, err_msg
        self.policy = policy
        self.policy_behavioural = copy.deepcopy(self.policy)

        err_msg = "[PG] data processor is None."
        assert data_processor is not None, err_msg
        self.data_processor = data_processor
        
        os.makedirs(directory, exist_ok=True)
        self.directory = directory

        # Other class' parameters
        self.ite = ite
        self.batch_size = batch_size
        self.verbose = verbose
        self.natural = natural
        self.baselines = baselines
        self.checkpoint_freq = checkpoint_freq
        self.n_jobs = n_jobs
        self.dim_action = self.env.action_dim
        self.dim_state = self.env.state_dim
        self.parallel_sampling = bool(self.n_jobs != 1)
        self.debug = debug
        self.seed = seed
        self.defensive_batchsize = defensive_batch_size
        self.evaluation_batch_size = evaluation_batch_size
        # Useful structures
        self.theta_history = np.zeros((self.ite, self.dim), dtype=np.float64)
        self.theta_behavioural_history = np.zeros((self.ite, self.dim), dtype = np.float64)
        self.time = 0
        self.performance_idx = np.zeros(ite, dtype=np.float64)
        self.estimated_gradient = np.zeros(ite, dtype=np.float64)
        self.best_theta = np.zeros(self.dim, dtype=np.float64)
        self.best_performance_theta = -np.inf
        self.sampler = TrajectorySampler(
            env=self.env, pol=self.policy, data_processor=self.data_processor
        )
        self.deterministic_curve = np.zeros(self.ite)
        self.behavioural_policies = []
        self.states = np.zeros((0, self.env.horizon, self.dim_state), dtype = np.float64)
        self.actions = np.zeros((0, self.env.horizon, self.dim_action), dtype = np.float64)
        self.reward_vector = np.zeros((0, self.env.horizon), dtype = np.float64)
        self.perf_vector = np.zeros(0, dtype = np.float64)

        # init the theta history
        self.theta_history[self.time, :] = copy.deepcopy(self.thetas)
        self.theta_behavioural_history[self.time, :] = copy.deepcopy(self.thetas_behavioural)

        # create the adam optimizers
        self.adam_optimizer = None
        if self.lr_strategy == "adam":
            self.adam_optimizer = Adam(alpha=self.lr)

        #initialize policies
        self.policy.set_parameters(copy.deepcopy(self.thetas))
        self.policy_behavioural.set_parameters(copy.deepcopy(self.thetas_behavioural))
        return

    def learn(self) -> None:
        """Learning function"""
        worker_dict = dict(
                        env=copy.deepcopy(self.env),
                        pol=copy.deepcopy(self.policy),
                        pol_b= copy.deepcopy(self.policy_behavioural),
                        dp=copy.deepcopy(self.data_processor),
                        params=copy.deepcopy(self.thetas),
                        params_b = copy.deepcopy(self.thetas_behavioural)
                        # seed=self.seed
                    )
                    
        # build the parallel functions
        delayed_functions = delayed(pg_sampling_worker_bpo)
        # parallel computation
        res = Parallel(n_jobs=self.n_jobs)(delayed_functions(**worker_dict, seed=self.seed+j) for j in range(self.batch_size))
        self.states = np.array([res[j][TrajectoryResults.StateList] for j in range(self.batch_size)])
        self.actions = np.array([res[j][TrajectoryResults.ActionList] for j in range(self.batch_size)])
        self.reward_vector = np.array([res[j][TrajectoryResults.RewList] for j in range(self.batch_size)])
        self.perf_vector = np.array([res[j][TrajectoryResults.PERF] for j in range(self.batch_size)])
        self.behavioural_policies.append(copy.deepcopy(self.policy_behavioural))


        for i in tqdm(range(self.ite)):
            if self.parallel_sampling:
              
                self.policy.set_parameters(copy.deepcopy(self.thetas))
                self.policy_behavioural.set_parameters(copy.deepcopy(self.thetas_behavioural))
                
                # evaluation of current target policy
                
                worker_dict_eval = dict(
                    env = copy.deepcopy(self.env),
                    pol=copy.deepcopy(self.policy),
                    dp=copy.deepcopy(self.data_processor),
                    params=copy.deepcopy(self.thetas),
                    # seed = self.seed
                )
                delayed_functions = delayed(pg_sampling_worker)

                # parallel computation
                res = Parallel(n_jobs=self.n_jobs)(delayed_functions(**worker_dict_eval, seed=self.seed+j+i*self.evaluation_batch_size) for j in range(self.evaluation_batch_size))
                perf_vector = np.zeros(self.evaluation_batch_size, dtype=np.float64)
                for j in range(self.evaluation_batch_size):
                    perf_vector[j] = res[j][TrajectoryResults.PERF]
                
                #print(f"perf_vector: {perf_vector}")
                self.performance_idx[i] = np.mean(perf_vector)
                # Update best rho
                self.update_best_theta(current_perf=self.performance_idx[i])
                
            else:
                raise NotImplementedError
                res = []
                for j in range(self.batch_size):
                    tmp_res = self.sampler.collect_trajectory_forBPO(params_target=copy.deepcopy(self.thetas), params_behavioural=copy.deepcopy(self.thetas_behavioural), seed=self.seed)
                    res.append(tmp_res)
            
            score_vector, weights_trajectories_log, _ = self.get_scores_and_weights(self.reward_vector, self.states, self.actions)
            
            weights_trajectories = np.exp(weights_trajectories_log)
            
            if self.estimator_type == "REINFORCE":
                grad_samples = self.perf_vector[:, np.newaxis] * np.sum(score_vector, axis=1)
                norm_vector = np.linalg.norm(grad_samples, axis = 1)
            elif self.estimator_type == "GPOMDP":
                grad_samples = self.get_samples_gpomdp(self.reward_vector, score_vector)
                norm_vector = np.linalg.norm(grad_samples, axis = 1)
                     
            coefficients = weights_trajectories * norm_vector
            
            if isinstance(self.policy, GaussianPolicy):
                print(f"Doing Closed form optimization")
                num = np.sum(coefficients[..., None] * np.sum(np.squeeze(self.actions, -1)[...,None] * self.states, axis = 1), axis = 0)
                out_prod = np.einsum('ntf,ntg->ntfg', self.states, self.states)
                den = np.sum(coefficients[...,None, None] * np.sum(out_prod, axis = 1),axis = 0)
                self.thetas_behavioural = num @ np.linalg.inv(den)
                self.policy_behavioural.set_parameters(copy.deepcopy(self.thetas_behavioural))
                print(f"Optimal behavioural policy parameters : {self.thetas_behavioural}")
            else:
                coefficients = torch.as_tensor(coefficients, dtype = torch.float64)
                loss = lambda coefficients, logps: - torch.mean(coefficients * torch.sum(logps, axis = 1), axis = 0)    
                #initialize behavioural policy with target policy
                self.policy_behavioural.set_parameters(copy.deepcopy(self.thetas))    
                
                max_iter = 500
                tol = 1000
                optimizer = optim.Adam(self.policy_behavioural.net.parameters(), lr=1e-4)
        
                # Set deterministic behavior for optimizer
                #torch.manual_seed(self.seed)
        
                for it in range(max_iter):
                    optimizer.zero_grad()
                    logprobs = self.policy_behavioural.compute_logprob_batch(self.states, self.actions)
                    loss_val = loss(coefficients, logprobs)
                    loss_val.backward()
                    grad_norm = sum(p.grad.norm().item() for p in self.policy_behavioural.net.parameters() if p.grad is not None)

                    if it % 20 == 0 : print(f"Iter {it:03d} | Loss: {loss_val.item():.4f} | Grad Norm: {grad_norm:.4f}")

                    optimizer.step()

                    if grad_norm < tol:
                        print("Converged.")
                        break
            
                self.thetas_behavioural = self.policy_behavioural.get_parameters()
                    
                    
            if self.parallel_sampling:
                    # parallel trajectory sampling
                    # prepare the parameters
                    worker_dict = dict(
                        env=copy.deepcopy(self.env),
                        pol=copy.deepcopy(self.policy),
                        pol_b= copy.deepcopy(self.policy_behavioural),
                        dp=copy.deepcopy(self.data_processor),
                        params=copy.deepcopy(self.thetas),
                        params_b = copy.deepcopy(self.thetas_behavioural)
                        # seed=self.seed
                    )
                    
                    # build the parallel functions
                    delayed_functions = delayed(pg_sampling_worker_bpo)
                    # parallel computation
                    res = Parallel(n_jobs=self.n_jobs)(delayed_functions(**worker_dict, seed=self.seed+j+i*self.batch_size) for j in range(self.batch_size))
            else:
                raise NotImplementedError
                res = []
                for j in range(self.batch_size):
                    tmp_res = self.sampler.collect_trajectory_forBPO(params_target=copy.deepcopy(self.thetas), params_behavioural= copy.deepcopy(self.thetas_behavioural), seed=self.seed)
                    res.append(tmp_res)
        
            score_vector, weights_trajectories_log, rolling_weights_log = self.process_batch(res)
            
            # Compute the estimated gradient            
            if self.estimator_type == "REINFORCE":
                weights_trajectories = np.exp(weights_trajectories_log)
                self.estimated_gradient = np.mean(
                    self.perf_vector[:, np.newaxis] * np.sum(score_vector, axis=1) * weights_trajectories[:, np.newaxis], axis=0)
            elif self.estimator_type == "GPOMDP":
                self.estimated_gradient = self.update_gpomdp_bpo(
                    reward_vector=self.reward_vector, score_vector=score_vector, rolling_weights_log = rolling_weights_log
                )
            else:
                err_msg = f"[PG] {self.estimator_type} has not been implemented yet!"
                raise NotImplementedError(err_msg)

            # Update parameters
            if self.lr_strategy == "constant":
                self.thetas = self.thetas + self.lr * self.estimated_gradient
            elif self.lr_strategy == "adam":
                adaptive_lr = self.adam_optimizer.next(self.estimated_gradient)
                self.thetas = self.thetas + adaptive_lr
            else:
                err_msg = f"[PG] {self.lr_strategy} not implemented yet!"
                raise NotImplementedError(err_msg)
            
            # Log
            if self.verbose:
                print("*" * 30)
                print(f"Step: {self.time}")
                print(f"Mean Performance: {self.performance_idx[self.time - 1]}")
                print(f"Estimated gradient: {self.estimated_gradient}")
                print(f"Parameter (new) values: {self.thetas}")
                print(f"Best performance so far: {self.best_performance_theta}")
                print(f"Best configuration so far: {self.best_theta}")
                print("*" * 30)

            # Checkpoint
            if self.time % self.checkpoint_freq == 0:
                self.save_results()

            # save theta history
            self.theta_history[self.time, :] = copy.deepcopy(self.thetas)
            self.theta_behavioural_history[self.time, :] = copy.deepcopy(self.thetas_behavioural)

            # time update
            self.time += 1

            # reduce the exploration factor of the policy
            self.policy.reduce_exploration() # ha senso lasciarlo ??? mi sa di no
            self.policy_behavioural.reduce_exploration()

        return

    
    def process_batch(
        self, res: TrajectoryResults
    ):
        self.behavioural_policies.append(copy.deepcopy(self.policy_behavioural))
        behavioural_policies = self.behavioural_policies
        
        perf_vector = np.zeros(self.batch_size, dtype=np.float64)
        reward_vector = np.zeros((self.batch_size, self.env.horizon), dtype=np.float64)            
        states = np.zeros((self.batch_size, self.env.horizon, self.dim_state), dtype = np.float64)
        actions = np.zeros((self.batch_size, self.env.horizon, self.dim_action), dtype = np.float64)
        
        for j in range(self.batch_size):
            perf_vector[j] = res[j][TrajectoryResults.PERF]
            reward_vector[j, :] = res[j][TrajectoryResults.RewList]
            states[j,:,:] = res[j][TrajectoryResults.StateList]
            actions[j,:,:] = res[j][TrajectoryResults.ActionList]
            
        perf_vector = np.concatenate((self.perf_vector, perf_vector), axis = 0)
        reward_vector = np.concatenate((self.reward_vector, reward_vector), axis = 0)
        states = np.concatenate((self.states, states), axis = 0)
        actions = np.concatenate((self.actions, actions), axis = 0)

        score_vector, log_weights_trajectories, log_weights_cumsum = self.get_scores_and_weights(reward_vector, states, actions)

        self.perf_vector = perf_vector
        self.reward_vector = reward_vector
        self.states = states
        self.actions = actions

        return score_vector,log_weights_trajectories,log_weights_cumsum

    def get_scores_and_weights(self, reward_vector, states, actions):
        mask = reward_vector
        score_vector = np.zeros((len(self.behavioural_policies) * self.batch_size, self.env.horizon, self.dim),
                                dtype=np.float64)
        for j in range(len(self.behavioural_policies)* self.batch_size):
            for k in range(self.env.horizon):
                score_vector[j, k, :] = self.policy.compute_score(states[j, k, :], actions[j, k, :])

        score_vector = score_vector * mask[..., None]

        logprobs_target = np.zeros((len(self.behavioural_policies) * self.batch_size, self.env.horizon), dtype=np.float64)
        logprobs_target = np.array([[self.policy.compute_logprob(states[j, k, :], actions[j, k, :]) 
                             for k in range(self.env.horizon)] 
                            for j in range(len(self.behavioural_policies) * self.batch_size)])
        
        logprobs_mis = np.zeros((len(self.behavioural_policies), self.batch_size * len(self.behavioural_policies), self.env.horizon), dtype = np.float64)
        for j in range(len(self.behavioural_policies)):
            #logprobs_mis[j,:,:] = behavioural_policies[j].compute_logprob_batch(states, actions).detach().numpy()
            logprobs_mis[j,:,:] = np.array([[self.behavioural_policies[j].compute_logprob(states[i, k, :], actions[i, k, :]) 
                             for k in range(self.env.horizon)] 
                            for i in range(len(self.behavioural_policies) * self.batch_size)])
        alpha = 1 / len(self.behavioural_policies)
        
        logprobs_target_sum = np.sum(logprobs_target, axis = 1)
        logprobs_mis_sum = np.sum(logprobs_mis, axis = 2)
        
        log_alpha = np.log(alpha)  # scalar
        log_mixture_probs_sum = logsumexp(
            log_alpha + logprobs_mis_sum,  
            axis=0  # sum over mixture components
        )  # → shape: [B]

        log_weights_trajectories = logprobs_target_sum - log_mixture_probs_sum
        
        logprobs_target_cms = np.cumsum(logprobs_target, axis = 1)
        logprobs_mis_cms = np.cumsum(logprobs_mis, axis = 2)
        
        log_mixture_probs_cms = logsumexp(
            log_alpha + logprobs_mis_cms,  # broadcasting over [K, B, H]
            axis=0  # sum over mixture components
        )  # → shape: [B, H]
        log_weights_cumsum = logprobs_target_cms - log_mixture_probs_cms  # → [B, H]
        return score_vector, log_weights_trajectories, log_weights_cumsum



    def get_samples_gpomdp(
        self, 
        reward_vector: np.ndarray,          # shape: (N, H)
        score_trajectory: np.ndarray        # shape: (N, H, m)
    ) -> np.ndarray:
        gamma = self.env.gamma
        horizon = self.env.horizon
        gamma_seq = (gamma * np.ones(horizon, dtype=np.float64)) ** np.arange(horizon)
        
        # === Mask: assume 0 reward marks episode end (true for CartPole)
        mask = (reward_vector != 0).astype(np.float64)  # shape: (N, H)
        
        # === Masked score trajectory (avoid using padded/terminated time steps)
        masked_scores = score_trajectory * mask[..., None]  # (N, H, m)
        
        # === Rolling scores (cumulative sum along time)
        rolling_scores = np.cumsum(masked_scores, axis=1)  # (N, H, m)
        
        # === Discounted reward
        discounted_rewards = reward_vector * gamma_seq[None, :]  # (N, H)
        
        # === Baseline: Peters per-time-step per-param vector baseline
        squared_scores = rolling_scores ** 2  # (N, H, m)
        b_num = np.sum(squared_scores * discounted_rewards[..., None], axis=0)  # (H, m)
        b_den = np.sum(squared_scores, axis=0) + 1e-10                          # (H, m)
        b = b_num / b_den                                                      # (H, m)

        # === Advantage: (r - b) * score
        adv = (discounted_rewards[..., None] - b[None, :, :]) * rolling_scores  # (N, H, m)

        # === Final GPOMDP sample estimate (sum over time)
        samples = np.sum(adv, axis=1)  # (N, m)

        return samples

        

    def update_gpomdp_bpo(
        self, reward_vector: np.array,
        score_vector: np.array,
        rolling_weights_log: np.array
        
    ) -> np.array:
        gamma = self.env.gamma
        horizon = self.env.horizon
        gamma_seq = (gamma * np.ones(horizon, dtype=np.float64)) ** (np.arange(horizon))
        rolling_scores = np.cumsum(score_vector, axis=1) 
        
        rolling_weights = np.exp(rolling_weights_log) 
         
        stabilizers = np.max(rolling_weights_log, axis = 0)
        
        
        if self.baselines == "avg":
            #only for cartpole 
            n_k = np.sum(reward_vector, axis = 0)
            n_k[n_k==0.] = 1
            b = np.sum(reward_vector, axis=0) / n_k
            reward_trajectory = (reward_vector - b[np.newaxis,...])[...,None] * rolling_scores * rolling_weights[..., np.newaxis]

        elif self.baselines == "peters":
            b = np.sum(rolling_scores ** 2 * reward_vector[...,None] * np.exp(2*(rolling_weights_log - stabilizers[None, ...]))[...,None], axis=0) / np.sum(rolling_scores ** 2 * np.exp(2*(rolling_weights_log - stabilizers[None,...]))[...,None] , axis=0)
            b[b != b] = 0
            reward_trajectory = (reward_vector[..., np.newaxis] - b[np.newaxis,...] ) * rolling_scores * rolling_weights[..., np.newaxis]


        else:
            reward_trajectory = reward_vector[..., None] * rolling_scores * rolling_weights[..., np.newaxis]

        self.estimated_gradient = np.mean(
            np.sum(gamma_seq[np.newaxis, :, np.newaxis]*reward_trajectory, axis = 1 ),
            axis = 0
        )
        
        # print(f"Reward Vector: {reward_vector}")
        # print(f"rolling_scores: {rolling_scores}")
        # print(f"weights_log: {weights_log}")
        #print(f"rolling_weighst: {rolling_weights}")
        print("DEBUG", rolling_scores.shape, b.shape, reward_trajectory.shape, reward_vector.shape, self.estimated_gradient.shape)
        print(f"estimated gradient: {self.estimated_gradient}")
        
        
        return self.estimated_gradient
        

    def update_best_theta(self, current_perf: np.float64) -> None:
        if self.best_theta is None or self.best_performance_theta <= current_perf:
            self.best_performance_theta = current_perf
            self.best_theta = copy.deepcopy(self.thetas)

            print("#" * 30)
            print("New best parameter configuration found")
            print(f"Performance: {self.best_performance_theta}")
            print(f"Parameter configuration: {self.best_theta}")
            print("#" * 30)
        return

    def save_results(self) -> None:
        if not self.debug:
            results = {
                "performance": np.array(self.performance_idx, dtype=float).tolist(),
                "best_theta": np.array(self.best_theta, dtype=float).tolist(),
                "gradient_history": np.array(self.estimated_gradient, dtype=np.float64).tolist(),
            }
        else:
            results = {
                "performance": np.array(self.performance_idx, dtype=float).tolist(),
                "best_theta": np.array(self.best_theta, dtype=float).tolist(),
                "thetas_history": np.array(self.theta_history, dtype=float).tolist(),
                "last_theta": np.array(self.thetas, dtype=float).tolist(),
                "best_perf": float(self.best_performance_theta),
            }

        # Save the json
        name = self.directory + "/results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
        return
    
   