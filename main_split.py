#todo --> use parameter from input
#todo --> generalize the entire class structure

# Libraries
import copy

from envs import *
from policies import GaussianPolicy, SplitGaussianPolicy, LinearPolicy, DeepGaussian
from algorithms import PolicyGradientSplit
from algorithms import PolicyGradient
from algorithms import PolicyGradientSplitAngles
from algorithms import PolicyGradientSplitVM
from data_processors import IdentityDataProcessor
from art import *
import torch
import torch.nn as nn
import json

"""Global Vars"""
# general
MODE = "learn_test"

# env_selection = ["lq", "swimmer", "cartpole"]
ENV = "lq"

# pol_selection = ["split_gaussian", "linear", "gaussian", "nn"]
POL = "split_gaussian"

alg_selection = ["pg", "split","split_angles","split_VM"]
ALG = alg_selection[3]

# environment
horizon = 75
gamma = 0.99
RENDER = False

# algorithm
DEBUG = False
NATURAL = False
ITE = 100
BATCH = 1000
N_JOBS_PARAM = 8
LR_STRATEGY = "constant"
BASELINE = "peters"

if ALG == "split":
    dir = f"/Users/Admin/OneDrive/Documenti/GitHub/learnRL/results/split/split_test_{ITE}_"
    ESTIMATOR = "GPOMDP"

if ALG== "pg":
    dir = f"/Users/Admin/OneDrive/Documenti/GitHub/learnRL/results/pg/pg_test_{ITE}_"
    ESTIMATOR = "GPOMDP"

if ALG=="split_angles":
    dir= f"/Users/Admin/OneDrive/Documenti/GitHub/learnRL/results/split_angles/split_angles_{ITE}_"
    ESTIMATOR = "GPOMDP"

if ALG=="split_VM":
    dir= f"/Users/Admin/OneDrive/Documenti/GitHub/learnRL/results/split_VM/split_VM_{ITE}_"
    ESTIMATOR = "GPOMDP"

if LR_STRATEGY == "adam":
    INIT_LR = 1e-3
    dir += "adam_0001_"
else:
    INIT_LR = 1e-3
    dir += "clr_00001_"

# test
test_ite = horizon
num_test = 10

"""Environment"""
if ENV == "lq":
    env_class = LQ
    env = LQ(horizon=horizon, gamma=gamma)
    dir += f"lq_{horizon}_"
elif ENV == "cartpole":
    env_class = ContCartPole
    env = ContCartPole(horizon=horizon, gamma=gamma, render=RENDER)
    dir += f"cartpole_{horizon}_"
else:
    raise NotImplementedError

s_dim = env.state_dim
a_dim = env.action_dim
MULTI_LINEAR = False

split_grid = np.linspace(env.max_pos, -env.max_pos, 50)
split_grid = np.append(split_grid, np.array(0))

"""Data Processor"""
dp = IdentityDataProcessor()

"""Policy"""
tot_params = s_dim * a_dim

if POL == "linear":
    pol = LinearPolicy(
        parameters=np.ones(tot_params),
        dim_state=s_dim,
        dim_action=a_dim,
        multi_linear=MULTI_LINEAR
    )
    tot_params = s_dim * a_dim
    dir += f"linear_policy_{tot_params}"
elif POL == "gaussian":
    pol = GaussianPolicy(
        parameters=np.ones(tot_params),
        dim_state=s_dim,
        dim_action=a_dim,
        std_dev=0.1,
        std_decay=0,
        std_min=1e-6,
        multi_linear=MULTI_LINEAR
    )
    dir += f"lingauss_policy_{tot_params}_var_01"
elif POL == "split_gaussian":
    pol = SplitGaussianPolicy(
        parameters=np.ones(tot_params),
        dim_state=s_dim,
        dim_action=a_dim,
        std_dev=0.1,
        std_decay=0,
        std_min=1e-6,
        multi_linear=MULTI_LINEAR,
        constant=True
    )
    dir += f"split_policy_{tot_params}_var_01"
elif POL == "nn":
    net = nn.Sequential(
        nn.Linear(s_dim, 16, bias=False),
        nn.Tanh(),
        nn.Linear(16, 8, bias=False),
        nn.Tanh(),
        nn.Linear(8, a_dim, bias=False)
    )
    model_desc = dict(
        layers_shape=[(s_dim, 16), (16, 8), (8, a_dim)]
    )

    pol = DeepGaussian(
        parameters=None,
        input_size=s_dim,
        output_size=a_dim,
        model=copy.deepcopy(net),
        model_desc=copy.deepcopy(model_desc),
        std_dev=0.1,
        std_decay=0,
        std_min=1e-6
    )
    tot_params = pol.tot_params
    dir += f"nn_policy_{tot_params}_var_01"
else:
    raise NotImplementedError


"""Algorithms"""
if ALG == "pg":
    alg_parameters = dict(
        lr=[INIT_LR],
        lr_strategy=LR_STRATEGY,
        estimator_type=ESTIMATOR,
        initial_theta=[0] * tot_params,
        ite=ITE,
        batch_size=BATCH,
        env=env,
        policy=pol,
        data_processor=dp,
        directory=dir,
        verbose=DEBUG,
        natural=NATURAL,
        checkpoint_freq=100,
        n_jobs=N_JOBS_PARAM,
        baselines=None,
    )
    alg = PolicyGradient(**alg_parameters)
if ALG=="split":
    alg_parameters = dict(
        lr=[INIT_LR],
        lr_strategy=LR_STRATEGY,
        estimator_type="GPOMDP",
        initial_theta=[0] * tot_params,
        ite=ITE,
        batch_size=BATCH,
        env=env,
        policy=pol,
        data_processor=dp,
        directory=dir,
        verbose=DEBUG,
        natural=NATURAL,
        checkpoint_freq=100,
        n_jobs=N_JOBS_PARAM,
        baselines=BASELINE,
        split_grid=split_grid
    )
    alg = PolicyGradientSplit(**alg_parameters)

if ALG=="split_angles":
    alg_parameters = dict(
        lr=[INIT_LR],
        lr_strategy=LR_STRATEGY,
        estimator_type="GPOMDP",
        initial_theta=[0] * tot_params,
        ite=ITE,
        batch_size=BATCH,
        env=env,
        policy=pol,
        data_processor=dp,
        directory=dir,
        verbose=DEBUG,
        natural=NATURAL,
        checkpoint_freq=100,
        n_jobs=N_JOBS_PARAM,
        baselines=BASELINE,
        split_grid=split_grid
    )
    alg = PolicyGradientSplitAngles(**alg_parameters)

if ALG=="split_VM":
    alg_parameters = dict(
        lr=[INIT_LR],
        lr_strategy=LR_STRATEGY,
        estimator_type="GPOMDP",
        initial_theta=[0] * tot_params,
        ite=ITE,
        batch_size=BATCH,
        env=env,
        policy=pol,
        data_processor=dp,
        directory=dir,
        verbose=DEBUG,
        natural=NATURAL,
        checkpoint_freq=100,
        n_jobs=N_JOBS_PARAM,
        baselines=BASELINE,
        split_grid=split_grid
    )
    alg = PolicyGradientSplitVM(**alg_parameters)

if __name__ == "__main__":
    # Learn phase
    print("GRIGLIA", split_grid)

    print(text2art("== LQ =="))
    if MODE in ["learn", "learn_test"]:
        print(text2art("Learn Start"))
        alg.learn()
        alg.save_results()
        print(alg.performance_idx)

    # Test phase
    # todo aggiusta il fatto dello 0 e vedi di mettere il std decay
    # todo to clip or not to clip
    if MODE in ["test", "learn_test"]:
        print(text2art("== TEST =="))
        env = env_class(horizon=horizon, gamma=gamma)
        if ALG == "pgpe":
            pol.set_parameters(thetas=alg.best_theta[0])
        else:
            pol.set_parameters(thetas=alg.best_theta)
            pol.std_dev = 0
        for _ in range(num_test):
            env.reset()
            state = env.state
            r = 0
            for i in range(test_ite):
                state, rew, _, _ = env.step(action=pol.draw_action(state))
                r += (gamma ** i) * rew
            print(f"PERFORMANCE: {r}")