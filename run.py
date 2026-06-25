# Libraries
import argparse
import copy
import datetime
from algorithms import (
    PolicyGradient,
    PolicyGradientSplit,
    ParameterPolicyGradientSplit,
    CLOLPlanning,
    PGPE,
)
from data_processors import IdentityDataProcessor
from envs import *
from policies import *
from art import *
import pickle
from common.utils import *
import random
import time
import json
import io

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--dir",
    help="Directory in which save the results.",
    type=str,
    default=""
)
parser.add_argument(
    "--ite",
    help="How many iterations the algorithm must do.",
    type=int,
    default=100
)
parser.add_argument(
    "--alg",
    help="The algorithm to use.",
    type=str,
    default="pg",
    choices=["pg", "agaps", "pgaps", "clol", "pgpe"]
)
parser.add_argument(
    "--estimator",
    help="The gradient estimator to use.",
    type=str,
    default="GPOMDP",
    choices=["REINFORCE", "GPOMDP"]
)
parser.add_argument(
    "--pol",
    help="The policy used.",
    type=str,
    default="split_gaussian",
    choices=["gaussian", "linear_gaussian", "linear", "split_gaussian", "nn", "deep_gaussian"]
)
parser.add_argument(
    "--std",
    help="The exploration amount.",
    type=float,
    default=0.1
)
parser.add_argument(
    "--env",
    help="The environment.",
    type=str,
    default="swimmer",
    choices=["swimmer", "half_cheetah", "ant", "lq", "minigolf", "mountain_car",
             "river", "cartpole", "hopper", "walker", "inverted_pendulum",
             "reacher", "pendulum", "dam", "ns_cartpole"]
)
parser.add_argument(
    "--horizon",
    help="The horizon amount.",
    type=int,
    default=200
)
parser.add_argument(
    "--gamma",
    help="The gamma amount.",
    type=float,
    default=1
)
parser.add_argument(
    "--lr",
    help="The lr amount.",
    type=float,
    default=1e-3
)
parser.add_argument(
    "--lr_strategy",
    help="The strategy employed for the lr.",
    type=str,
    default="constant",
    choices=["adam", "constant"]
)
parser.add_argument(
    "--batch",
    help="The batch size.",
    type=int,
    default=100
)
parser.add_argument(
    "--clip",
    help="Whether to clip the action in the environment.",
    type=int,
    default=1,
    choices=[0, 1]
)
parser.add_argument(
    "--n_trials",
    help="How many runs of the same experiment to perform.",
    type=int,
    default=1
)
parser.add_argument(
    "--lq_state_dim",
    help="State dimension for the LQR environment.",
    type=int,
    default=1
)
parser.add_argument(
    "--lq_action_dim",
    help="Action dimension for the LQR environment.",
    type=int,
    default=1
)
parser.add_argument(
    "--verbose",
    help="Print debug information.",
    type=int,
    default=0
)
parser.add_argument(
    "--baseline",
    help="The baseline choosen.",
    type=str,
    default="peters",
    choices=["none", "avg", "peters"]
)
parser.add_argument(
    "--alpha",
    help="The split penalization coefficient (AGAPS/PGAPS).",
    type=float,
    default=1e-1,
)
parser.add_argument(
    "--max_splits",
    help="Maximum number of division (AGAPS/PGAPS).",
    type=int,
    default=10,
)
parser.add_argument(
    "--deterministic",
    help="Deterministic piecewise policy.",
    type=bool,
    default=False,
)
parser.add_argument(
    "--linear",
    help="Linear piecewise policy.",
    type=bool,
    default=False,
)
parser.add_argument(
    "--planning_horizon",
    help="The planning horizon (CLOL).",
    type=int,
    default=1
)
parser.add_argument(
    "--persistence",
    help="Use action persistence (CLOL).",
    default=False,
    action='store_true'
)
parser.add_argument(
    "--n_jobs",
    help="Number of parallel jobs for sampling and computation.",
    type=int,
    default=1,
)
parser.add_argument(
    "--animate",
    help="Render the agent.",
    default=False,
    action='store_true'
)
parser.add_argument(
    "--debug",
    help="Output the whole set of information.",
    default=False,
    action='store_true'
)
parser.add_argument(
    "--evaluate",
    help="Evaluate the policy without training.",
    default=False,
    action='store_true'
)
parser.add_argument(
    "--starting_seed",
    help="Starting seed value.",
    type=int,
    default=0
)
parser.add_argument(
    "--noise",
    help="Noise injected in the MDP as std.",
    type=float,
    default=0.0
)
parser.add_argument(
    "--force",
    help="Only for ns_cartpole, define the force scale.",
    type=float,
    default=10.0
)
parser.add_argument(
    "--friction",
    help="Only for ns_cartpole, define the fixed friction coefficient.",
    type=float,
    default=0.01
)

args = parser.parse_args()

if args.std < 1:
    string_var = str(args.std).replace(".", "")
else:
    string_var = str(int(args.std))

# Build
base_dir = args.dir
base_dir += "_" + datetime.datetime.now().strftime("%m_%d-%H_%M_")

total_time = np.zeros(args.n_trials)

for trial in range(args.n_trials):
    seed = trial + args.starting_seed

    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    """Directory name"""
    if args.alg == "clol":
        dir_name = f"CLOL_Planning_{args.ite}_{args.env}_{args.horizon}_{args.planning_horizon}_{str(args.gamma).replace('.', '')}_{args.lr_strategy}_"
    elif args.alg == "pgpe":
        dir_name = f"PGPE_{args.ite}_{args.env}_{args.horizon}_{str(args.gamma).replace('.', '')}_{args.lr_strategy}_"
    else:
        dir_name = f"{args.alg.upper()}_{args.ite}_{args.env}_{args.horizon}_{str(args.gamma).replace('.', '')}_{args.lr_strategy}_"
    dir_name += f"{str(args.lr).replace('.', '')}_{args.batch}_"

    if args.clip:
        dir_name += "clip_"
    else:
        dir_name += "noclip_"

    if args.env == "lq":
        dir_name += f"dS_{args.lq_state_dim}_dA_{args.lq_action_dim}_"

    if args.env == "ns_cartpole":
        dir_name += f"force_{args.force}_mu_p_{args.friction}_"

    if args.alg == "pg":
        if args.pol == "linear":
            args.pol = "gaussian"
        elif args.pol == "nn":
            args.pol = "deep_gaussian"

    """Environment"""
    MULTI_LINEAR = False

    if args.env == "swimmer":
        env_class = Swimmer
        env = Swimmer(horizon=args.horizon, gamma=args.gamma, clip=bool(args.clip))
        MULTI_LINEAR = True
    elif args.env == "half_cheetah":
        env_class = HalfCheetah
        env = HalfCheetah(horizon=args.horizon, gamma=args.gamma, clip=bool(args.clip))
        MULTI_LINEAR = True
    elif args.env == "ant":
        env_class = Ant
        env = Ant(horizon=args.horizon, gamma=args.gamma, clip=bool(args.clip))
        MULTI_LINEAR = True
    elif args.env == "ns_cartpole":
        env_class = CartPoleEnv
        env = CartPoleEnv(horizon=args.horizon, gamma=args.gamma, mu_p=args.friction)
        MULTI_LINEAR = True
    elif args.env == "hopper":
        env_class = Hopper
        env = Hopper(horizon=args.horizon, gamma=args.gamma, clip=bool(args.clip))
    elif args.env == "reacher":
        env_class = Reacher
        env = Reacher(horizon=args.horizon, gamma=args.gamma, clip=bool(args.clip))
        MULTI_LINEAR = True
    elif args.env == "inverted_pendulum":
        env_class = InvertedPendulum
        env = InvertedPendulum(horizon=args.horizon, gamma=args.gamma, clip=bool(args.clip))
    elif args.env == "lq":
        env_class = LQ
        env = LQ(horizon=args.horizon, gamma=args.gamma, action_dim=args.lq_action_dim, state_dim=args.lq_state_dim, noise=args.noise)
        MULTI_LINEAR = bool(args.lq_action_dim > 1)
    elif args.env == "minigolf":
        env_class = MiniGolf
        env = MiniGolf(horizon=args.horizon, gamma=args.gamma)
    elif args.env == "mountain_car":
        env_class = Continuous_MountainCarEnv
        env = Continuous_MountainCarEnv(horizon=args.horizon, gamma=args.gamma)
        MULTI_LINEAR = True
    elif args.env == "river":
        env_class = RiverSwimContinuous
        env = RiverSwimContinuous(horizon=args.horizon, gamma=args.gamma)
    elif args.env == "cartpole":
        env_class = ContCartPole
        env = ContCartPole(horizon=args.horizon, gamma=args.gamma)
    elif args.env == "pendulum":
        env_class = PendulumEnv
        env = PendulumEnv(horizon=args.horizon, gamma=args.gamma, clip=bool(args.clip))
    elif args.env == "dam":
        env_class = Dam
        env = Dam(horizon=args.horizon, gamma=args.gamma)
    else:
        raise ValueError(f"Invalid env name.")

    s_dim = env.state_dim
    if args.persistence:
        a_dim = env.action_dim
        dir_name += f"_persistence"
    else:
        # planning_horizon defaults to 1, so this is a no-op for non-planning algorithms.
        a_dim = env.action_dim * args.planning_horizon

    """Data Processor"""
    dp = IdentityDataProcessor()
        
    """Policy"""
    if args.pol == "linear":
        tot_params = s_dim * a_dim
        pol = LinearPolicy(
            parameters=np.zeros(tot_params),
            dim_state=s_dim,
            dim_action=a_dim,
            multi_linear=MULTI_LINEAR
        )
    elif args.pol in ["gaussian", "linear_gaussian"]:
        tot_params = s_dim * a_dim
        pol = GaussianPolicy(
            parameters=np.zeros(tot_params),
            dim_state=s_dim,
            dim_action=a_dim,
            std_dev=args.std,
            std_decay=0,
            std_min=1e-6,
            multi_linear=MULTI_LINEAR
        )
    elif args.pol == "split_gaussian":
        # Linear leaves carry a full (a_dim, s_dim) gain per region; constant
        # leaves carry an action-sized mean.
        tot_params = a_dim * s_dim if args.linear else a_dim
        pol = SplitGaussianPolicy(
            parameters=np.zeros(tot_params),
            dim_state=s_dim,
            dim_action=a_dim,
            std_dev=args.std,
            std_decay=0,
            std_min=1e-6,
            deterministic=args.deterministic,
            linear=args.linear
        )
    elif args.pol in ["nn", "deep_gaussian"]:
        net = nn.Sequential(
            nn.Linear(s_dim, 50, bias=False),
            nn.Tanh(),
            nn.Linear(50, 25, bias=False),
            nn.Tanh(),
            nn.Linear(25, a_dim, bias=False),
            nn.Tanh()
        )
        model_desc = dict(
            layers_shape=[(s_dim, 50), (50, 25), (25, a_dim)]
        )
        if args.pol == "nn":
            pol = NeuralNetworkPolicy(
                parameters=None,
                input_size=s_dim,
                output_size=a_dim,
                model=copy.deepcopy(net),
                model_desc=copy.deepcopy(model_desc)
            )
        elif args.pol == "deep_gaussian":
            pol = DeepGaussianPolicy(
                parameters=None,
                input_size=s_dim,
                output_size=a_dim,
                model=copy.deepcopy(net),
                model_desc=copy.deepcopy(model_desc),
                std_dev=args.std,
                std_decay=0,
                std_min=1e-6
            )
        tot_params = pol.tot_params
    else:
        raise ValueError(f"Invalid policy name.")

    dir_name += f"_{args.pol}_{tot_params}_std_{string_var}"
    if args.alg in ["agaps", "pgaps"]:
        dir_name += f"_alpha_{str(args.alpha).replace('.', '')}"
    if args.linear:
        dir_name += f"_linear"
    dir_name += f"_noise_{str(args.noise).replace('.', '')}"
    dir_render = base_dir + dir_name + "/render.gif"
    time_dir = base_dir + dir_name + "/time.json"
    dir_name = base_dir + dir_name + "/" + f"trial_{seed}"

    """Algorithms"""
    if args.alg == "pg":
        alg_parameters = dict(
            lr=[args.lr],
            lr_strategy=args.lr_strategy,
            estimator_type=args.estimator,
            initial_theta=pol.parameters,
            ite=args.ite,
            batch_size=args.batch,
            env=env,
            policy=pol,
            data_processor=dp,
            directory=dir_name,
            verbose=args.verbose,
            checkpoint_freq=50,
            baselines=args.baseline,
            debug=args.debug,
            n_jobs=args.n_jobs,
            seed=seed
        )
        alg = PolicyGradient(**alg_parameters)
    elif args.alg == "agaps":
        alg_parameters = dict(
            lr=[args.lr],
            lr_strategy=args.lr_strategy,
            estimator_type=args.estimator,
            initial_theta=pol.parameters,
            ite=args.ite,
            batch_size=args.batch,
            env=env,
            policy=pol,
            data_processor=dp,
            directory=dir_name,
            verbose=args.verbose,
            checkpoint_freq=50,
            n_jobs=args.n_jobs,
            baselines=args.baseline,
            alpha=args.alpha,
            max_splits=args.max_splits,
            seed=seed
        )
        alg = PolicyGradientSplit(**alg_parameters)
    elif args.alg == "pgaps":
        alg_parameters = dict(
            lr=[args.lr],
            lr_strategy=args.lr_strategy,
            estimator_type=args.estimator,
            initial_rho=pol.parameters,
            ite=args.ite,
            batch_size=args.batch,
            env=env,
            policy=pol,
            data_processor=dp,
            directory=dir_name,
            verbose=args.verbose,
            checkpoint_freq=50,
            n_jobs=1,
            baselines=args.baseline,
            alpha=args.alpha,
            max_splits=args.max_splits
        )
        alg = ParameterPolicyGradientSplit(**alg_parameters)
    elif args.alg == "clol":
        alg_parameters = dict(
            lr=[args.lr],
            lr_strategy=args.lr_strategy,
            estimator_type=args.estimator,
            initial_theta=pol.parameters,
            ite=args.ite,
            batch_size=args.batch,
            env=env,
            policy=pol,
            data_processor=dp,
            directory=dir_name,
            verbose=args.verbose,
            checkpoint_freq=50,
            baselines=args.baseline,
            planning_horizon=args.planning_horizon,
            debug=args.debug,
            n_jobs=args.n_jobs,
            persistence=args.persistence,
            seed=seed
        )
        alg = CLOLPlanning(**alg_parameters)
    elif args.alg == "pgpe":
        init_mean = np.zeros(tot_params, dtype=np.float64).reshape(-1)
        init_std = np.ones_like(init_mean, dtype=np.float64) * args.std
        initial_rho = np.array([init_mean, init_std], dtype=np.float64)

        alg_parameters = dict(
            lr=[args.lr],
            initial_rho=initial_rho,
            ite=args.ite,
            batch_size=args.batch,
            episodes_per_theta=1,
            env=env,
            policy=pol,
            data_processor=dp,
            directory=dir_name,
            verbose=args.verbose,
            checkpoint_freq=50,
            lr_strategy=args.lr_strategy,
            n_jobs_param=args.n_jobs,
            n_jobs_traj=1
        )
        alg = PGPE(**alg_parameters)
    else:
        raise ValueError("Invalid algorithm name.")

    print(text2art(f"== {args.alg} TEST on {args.env} =="))
    print(text2art(f"Trial {seed}"))
    print(args)
    print(text2art("Learn Start"))
    alg.learn()
    end_time = time.time()
    alg.save_results()
    # save the policy parameters for each trial
    if hasattr(alg, "best_theta"):
        np.save(f"{dir_name}/policy_params", alg.best_theta)
    print(alg.performance_idx)

    time_trial = end_time - start_time
    total_time[trial] = time_trial

time_res = {"time": np.array(total_time, dtype=float).tolist()}
with io.open(time_dir, 'w', encoding='utf-8') as f:
    f.write(json.dumps(time_res, ensure_ascii=False, indent=4))
    f.close()

if args.animate:
    eval_env = env_class(horizon=args.horizon, gamma=args.gamma, render_mode="rgb_array")
    frames = evaluate_planning(
        eval_env, pol, num_episodes=1, horizon=args.horizon,
        persistence=args.persistence, planning_horizon=args.planning_horizon, dir=dir_name
    )
    imageio.mimsave(dir_render, frames, duration=33)
