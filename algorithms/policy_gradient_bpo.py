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
class PolicyGradientBpo:
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
        
        # fare error message e inizializzazione di : K=num_behav -  lista batches [100, ..] - defensive batch 

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

        # init the theta history
        self.theta_history[self.time, :] = copy.deepcopy(self.thetas)
        self.theta_behavioural_history[self.time, :] = copy.deepcopy(self.thetas_behavioural)

        # create the adam optimizers
        self.adam_optimizer = None
        if self.lr_strategy == "adam":
            self.adam_optimizer = Adam(alpha=self.lr)
        return

    def learn(self) -> None:
        """Learning function"""
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


                
                
                
                
                
                # parallel trajectory sampling
                # prepare the parameters                
                worker_dict = dict(
                    env=copy.deepcopy(self.env),
                    pol=copy.deepcopy(self.policy),
                    pol_b = copy.deepcopy(self.policy_behavioural),
                    dp=copy.deepcopy(self.data_processor),
                    params=copy.deepcopy(self.thetas),
                    params_b = copy.deepcopy(self.thetas_behavioural),
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
                    tmp_res = self.sampler.collect_trajectory_forBPO(params_target=copy.deepcopy(self.thetas), params_behavioural=copy.deepcopy(self.thetas_behavioural), seed=self.seed)
                    res.append(tmp_res)
            

            
            perf_vector = np.zeros(self.batch_size, dtype=np.float64)
            score_vector = np.zeros((self.batch_size, self.env.horizon, self.dim),
                                    dtype=np.float64)
            reward_vector = np.zeros((self.batch_size, self.env.horizon), dtype=np.float64)
            norm_vector = np.zeros(self.batch_size, dtype = np.float64)
            logprobs_vector_t = np.zeros((self.batch_size, self.env.horizon), dtype = np.float64)
            logprobs_vector_b = np.zeros((self.batch_size, self.env.horizon),  dtype = np.float64)
            states = np.zeros((self.batch_size, self.env.horizon, self.dim_state), dtype = np.float64)
            actions = np.zeros((self.batch_size, self.env.horizon, self.dim_action), dtype = np.float64)
            
            
            for j in range(self.batch_size):
                perf_vector[j] = res[j][TrajectoryResults.PERF]
                reward_vector[j, :] = res[j][TrajectoryResults.RewList]
                score_vector[j, :, :] = res[j][TrajectoryResults.ScoreList]
                logprobs_vector_t[j, :] = res[j][TrajectoryResults.Logprob_target]
                logprobs_vector_b[j, :] = res[j][TrajectoryResults.Logprob_behavioural]
                states[j,:,:] = res[j][TrajectoryResults.StateList]
                actions[j,:,:] = res[j][TrajectoryResults.ActionList]
                
                if self.estimator_type == "REINFORCE":
                    norm_vector[j] = np.linalg.norm (perf_vector[j] * np.sum(score_vector[j,:,:], axis=0)) # controllare giuste le dimensioni
                elif self.estimator_type == "GPOMDP":
                    discounted_rewards = reward_vector[j, :] * (self.env.gamma ** np.arange(self.env.horizon))  # shape (T,)
                    cumulative_scores = np.cumsum(score_vector[j, :, :], axis=0)  # shape (T, D)
                    norm_vector[j] = np.linalg.norm( np.sum(discounted_rewards[:, None] * cumulative_scores, axis=0) ) # shape (1)
            
            
            logprobs_t_sum = np.sum(logprobs_vector_t, axis = 1)
            logprobs_b_sum = np.sum(logprobs_vector_b, axis = 1)
            weights_trajectories = np.exp(logprobs_t_sum - logprobs_b_sum)
         
            coefficients = weights_trajectories * norm_vector
            #coefficients = coefficients - np.mean(coefficients)
            #print(f"coefficents for behavioural policy optimization {coefficients}")
            
            if isinstance(self.policy, GaussianPolicy):
                print(f"Doing Closed form optimization")
                num = np.sum(coefficients[..., None] * np.sum(np.squeeze(actions, -1)[...,None] * states, axis = 1), axis = 0)
                out_prod = np.einsum('ntf,ntg->ntfg', states, states)
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
                    #logprobs = torch.as_tensor([self.policy_behavioural.compute_logprob(state_, action_) for state_, action_ in zip(states, actions)])
                    logprobs = self.policy_behavioural.compute_logprob_batch(states, actions)
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
                    
                    worker_dict_defensive = dict(
                        env = copy.deepcopy(self.env),
                        pol =copy.deepcopy(self.policy_behavioural), 
                        pol_b=copy.deepcopy(self.policy), # sono scambiati a posta perché voglio samplare dalla target
                        dp=copy.deepcopy(self.data_processor),
                        params = copy.deepcopy(self.thetas_behavioural),
                        params_b = copy.deepcopy(self.thetas) 
                    )

                    # build the parallel functions
                    delayed_functions = delayed(pg_sampling_worker_bpo)
                    delayed_functions_defensive = delayed(pg_sampling_worker_bpo)

                    # parallel computation
                    res = Parallel(n_jobs=self.n_jobs)(delayed_functions(**worker_dict, seed=self.seed+j+i*self.batch_size) for j in range(self.batch_size))
                    res_d = Parallel(n_jobs=self.n_jobs)(delayed_functions_defensive(**worker_dict_defensive, seed=self.seed+j+i*self.defensive_batchsize) for j in range(self.batch_size,self.batch_size + self.defensive_batchsize))
            else:
                raise NotImplementedError
                res = []
                for j in range(self.batch_size):
                    tmp_res = self.sampler.collect_trajectory_forBPO(params_target=copy.deepcopy(self.thetas), params_behavioural= copy.deepcopy(self.thetas_behavioural), seed=self.seed)
                    res.append(tmp_res)
        
            # Update performance
            perf_vector = np.zeros(self.batch_size, dtype=np.float64)
            score_vector = np.zeros((self.batch_size, self.env.horizon, self.dim),
                                    dtype=np.float64)
            reward_vector = np.zeros((self.batch_size, self.env.horizon), dtype=np.float64)
            logprobs_vector_t = np.zeros((self.batch_size, self.env.horizon), dtype = np.float64)
            logprobs_vector_b = np.zeros((self.batch_size, self.env.horizon),  dtype = np.float64)
            
            perf_vector_d = np.zeros(self.defensive_batchsize, dtype=np.float64)
            score_vector_d = np.zeros((self.defensive_batchsize, self.env.horizon, self.dim),
                                    dtype=np.float64)
            reward_vector_d = np.zeros((self.defensive_batchsize, self.env.horizon), dtype=np.float64)
            logprobs_vector_t_d = np.zeros((self.defensive_batchsize, self.env.horizon), dtype = np.float64)
            logprobs_vector_b_d= np.zeros((self.defensive_batchsize, self.env.horizon),  dtype = np.float64)
            
            
            for j in range(self.batch_size):
                perf_vector[j] = res[j][TrajectoryResults.PERF]
                reward_vector[j, :] = res[j][TrajectoryResults.RewList]
                score_vector[j, :, :] = res[j][TrajectoryResults.ScoreList]
                logprobs_vector_t[j, :] = res[j][TrajectoryResults.Logprob_target]
                logprobs_vector_b[j, :] = res[j][TrajectoryResults.Logprob_behavioural]
                
            for j in range(self.defensive_batchsize):
                perf_vector_d[j] = res_d[j][TrajectoryResults.PERF]
                reward_vector_d[j, :] = res_d[j][TrajectoryResults.RewList]
                score_vector_d[j, :, :] = res_d[j][TrajectoryResults.ScoreList]
                logprobs_vector_t_d[j, :] = res_d[j][TrajectoryResults.Logprob_behavioural] # sono scambiati a posta perché voglio samplare dalla target
                logprobs_vector_b_d[j, :] = res_d[j][TrajectoryResults.Logprob_target]
                
            perf_vector = np.concatenate((perf_vector, perf_vector_d), axis = 0)
            reward_vector = np.concatenate((reward_vector, reward_vector_d), axis = 0)
            score_vector = np.concatenate((score_vector, score_vector_d), axis = 0)
            
            alpha1 = self.defensive_batchsize / (self.batch_size + self.defensive_batchsize)
            alpha2 = self.batch_size / (self.batch_size + self.defensive_batchsize)
            
            logprobs_t_sum = np.sum(logprobs_vector_t, axis = 1)
            logprobs_t_cumsum = np.cumsum(logprobs_vector_t, axis = 1)
            #logprobs_b_sum = np.sum(logprobs_vector_b, axis = 1)
            #logprobs_b_cumsum = np.cumsum(logprobs_vector_b, axis = 1)
            
            logprobs_t_d_sum = np.sum(logprobs_vector_t_d, axis = 1)
            logprobs_t_d_cumsum = np.cumsum(logprobs_vector_t_d, axis = 1)
            #logprobs_b_d_sum = np.sum(logprobs_vector_b_d, axis = 1)
            #logprobs_b_d_cumsum = np.cumsum(logprobs_vector_b_d, axis = 1)
            
            num_sum = np.concatenate((logprobs_t_sum, logprobs_t_d_sum), axis = 0)
            #den_sum = np.sum(np.concatenate((np.log(alpha1 * np.exp(logprobs_vector_t) + alpha2 *np.exp(logprobs_vector_b)), np.log(alpha1 * np.exp(logprobs_vector_t_d) + alpha2 * np.exp(logprobs_vector_b_d))), axis = 0 ), axis = 1)
            
            den_sum_part1 = logsumexp(np.stack([np.log(alpha1) + logprobs_vector_t, np.log(alpha2) + logprobs_vector_b], axis=0),axis=0)
            den_sum_part2 = logsumexp(np.stack([np.log(alpha1) + logprobs_vector_t_d,np.log(alpha2) + logprobs_vector_b_d], axis=0),axis=0)
            den_sum = np.sum(np.concatenate((den_sum_part1, den_sum_part2), axis=0), axis=1)
            den_cumsum = np.cumsum(np.concatenate((den_sum_part1, den_sum_part2), axis=0), axis=1)




            num_cumsum = np.concatenate((logprobs_t_cumsum, logprobs_t_d_cumsum), axis = 0)
            #den_cumsum = np.cumsum(np.concatenate((np.log(alpha1 * np.exp(logprobs_vector_t) + alpha2 *np.exp(logprobs_vector_b)), np.log(alpha1 * np.exp(logprobs_vector_t_d) + alpha2 * np.exp(logprobs_vector_b_d))), axis = 0 ), axis = 1)

            '''
            weights_log  = np.log(np.exp(logprobs_vector_t ))- np.log(alpha1 * np.exp(logprobs_vector_t) + alpha2 *np.exp(logprobs_vector_b) )
            weights_log_d = np.log(np.exp(logprobs_vector_t_d ))- np.log(alpha1 * np.exp(logprobs_vector_t_d) + alpha2 * np.exp(logprobs_vector_b_d))
            weights_log = np.concatenate((weights_log, weights_log_d), axis = 0)
            '''
            
            #print(f"weights log after concatenation {weights_log}")
            # Compute the estimated gradient
            #weights_log  = logprobs_vector_t - logprobs_vector_b
            #print(f"weights log : {weights_log}")
            
            if self.estimator_type == "REINFORCE":
                weights_trajectories_log = num_sum - den_sum
                weights_trajectories = np.exp(weights_trajectories_log)
                self.estimated_gradient = np.mean(
                    perf_vector[:, np.newaxis] * np.sum(score_vector, axis=1) * weights_trajectories[:, np.newaxis], axis=0)
            elif self.estimator_type == "GPOMDP":
                rolling_weights_log = num_cumsum - den_cumsum
                self.estimated_gradient = self.update_gpomdp_bpo(
                    reward_vector=reward_vector, score_vector=score_vector, rolling_weights_log = rolling_weights_log
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

    def update_gpomdp(
            self, reward_vector: np.array,
            score_trajectory: np.array
    ) -> np.array:
        gamma = self.env.gamma
        horizon = self.env.horizon
        gamma_seq = (gamma * np.ones(horizon, dtype=np.float64)) ** (np.arange(horizon))
        rolling_scores = np.cumsum(score_trajectory, axis=1) + 1e-10

        
        if self.baselines == "avg":
            b = np.mean(reward_vector[...,None], axis=0)
        elif self.baselines == "peters":
            b = np.sum(rolling_scores ** 2 * reward_vector[...,None], axis=0) / np.sum(rolling_scores ** 2, axis=0)
        else:
            b = np.zeros(1)

        reward_trajectory = (reward_vector[...,None] - b[np.newaxis,...]) * rolling_scores

        self.estimated_gradient = np.mean(
            np.sum(gamma_seq[:, np.newaxis] * reward_trajectory, axis=1),
            axis=0)

        #print("DEBUG", rolling_scores.shape, b.shape, reward_trajectory.shape, reward_vector.shape, self.estimated_gradient.shape)
        return self.estimated_gradient

    def update_gpomdp_bpo(
        self, reward_vector: np.array,
        score_vector: np.array,
        rolling_weights_log: np.array
        
    ) -> np.array:
        gamma = self.env.gamma
        horizon = self.env.horizon
        gamma_seq = (gamma * np.ones(horizon, dtype=np.float64)) ** (np.arange(horizon))
        rolling_scores = np.cumsum(score_vector, axis=1) + 1e-10
        rolling_weights = np.exp(rolling_weights_log) 
        
        stabilizers = np.max(rolling_weights_log, axis = 0)
        
        
        if self.baselines == "avg":
            b = np.mean(reward_vector[...,None], axis=0)
        elif self.baselines == "peters":
            b = np.sum(rolling_scores ** 2 * reward_vector[...,None] * np.exp(2*(rolling_weights_log - stabilizers[None, ...]))[...,None], axis=0) / np.sum(rolling_scores ** 2 * np.exp(2*(rolling_weights_log - stabilizers[None,...]))[...,None] , axis=0)
            b[b != b] = 0

        else:
            b = np.zeros(1)
        
        reward_trajectory = (reward_vector[..., np.newaxis] - b[np.newaxis,...] ) * rolling_scores * rolling_weights[..., np.newaxis]
        self.estimated_gradient = np.mean(
            np.sum(gamma_seq[:, np.newaxis]*reward_trajectory, axis = 1 ),
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
    
   