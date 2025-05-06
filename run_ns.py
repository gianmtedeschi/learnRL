# Libraries
import argparse
import datetime
from algorithms import PolicyGradientBpo
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
    default=5
)
parser.add_argument(
    "--estimator",
    help="The algorithm to use.",
    type=str,
    default="GPOMDP",
    choices=["REINFORCE", "GPOMDP"]
)
parser.add_argument(
    "--pol",
    help="The policy used.",
    type=str,
    default="deep_gaussian",
    choices=["linear_gaussian", "linear", "nn", "deep_gaussian"]
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
    default="pendulum",
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
    default=0,
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
    "--n_jobs",
    help="Number of workers.",
    type=int,
    default=1
)
parser.add_argument(
    "--animate",
    help="Render the agent",
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
    "--evaluate",
    help="Evaluate the policy without training.",
    default=False,
    action='store_true'
)
parser.add_argument(
    "--force",
    help="Only for cartpole, define the force scale.",
    type=float,
    default=10.0
)
parser.add_argument(
    "--friction",
    help="Only for cartpole, define the fixed friction coefficient.",
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

for i in range(args.n_trials):
    i += args.starting_seed

    start_time = time.time()
    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)

    dir_name = f"PG_{args.ite}_{args.env}_{args.horizon}_{str(args.gamma).replace('.', '')}_{args.lr_strategy}_"
    dir_name += f"{str(args.lr).replace('.', '')}_{args.batch}_"
    
    if args.clip:
        dir_name += "clip_"
    else:       dir_name += "noclip_"

    if args.env == "lq":
        dir_name += f"dS_{args.lq_state_dim}_dA_{args.lq_action_dim}_"
    
    if args.env == "ns_cartpole" or args.env == "ns_cartpole_2":
        dir_name += f"force_{args.force}_mu_p_{args.friction}_"
    
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
    elif args.env == "river":
        env_class = RiverSwimContinuous
        env = RiverSwimContinuous(horizon=args.horizon, gamma=args.gamma)
    elif args.env == "cartpole":
        env_class = ContCartPole
        env = ContCartPole(horizon=args.horizon, gamma=args.gamma)
    elif args.env == "pendulum":
        env_class = PendulumEnv
        env = PendulumEnv(horizon=args.horizon, gamma=args.gamma)
    elif args.env == "dam":
        env_class = Dam
        env = Dam(horizon=args.horizon, gamma=args.gamma)
    else:
        raise ValueError(f"Invalid env name.")

    s_dim = env.state_dim
    a_dim = env.action_dim

    """Data Processor"""
    dp = IdentityDataProcessor()

    """Policy"""
    if args.pol == "linear_gaussian":
        tot_params = s_dim * a_dim
        pol = GaussianPolicy(
            parameters=np.ones(tot_params),
            dim_state=s_dim,
            dim_action=a_dim,
            std_dev=args.std,
            std_decay=0,
            std_min=1e-6,
            multi_linear=MULTI_LINEAR
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
            # layers_shape=[(s_dim, a_dim)]
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
                # parameters=np.load('/Users/gianmarcotedeschi/Projects/learnRL/results_swingup/_03_21-17_32_PG_300_ns_cartpole_2_2000_10_adam_0005_100_noclip_force_10.0_mu_p_0.001__deep_gaussian_1475_std_1_noise_00/trial_0/policy_params.npy'),
                parameters=None,
                input_size=s_dim,
                output_size=a_dim,
                model=copy.deepcopy(net),
                model_desc=copy.deepcopy(model_desc),
                std_dev=args.std,
                std_decay=0,
                std_min=1e-6
            )
        else:
            raise ValueError("Invalid nn policy name.")
        tot_params = pol.tot_params
    else:
        raise ValueError(f"Invalid policy name.")

    dir_name += f"_{args.pol}_{tot_params}_std_{string_var}_noise_{str(args.noise).replace('.', '')}"
    dir_render = base_dir + dir_name + "/render.gif" 
    time_dir = base_dir + dir_name + "/time.json"
    dir_name = base_dir + dir_name + "/" + f"trial_{i}"

    """Algorithms"""
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
        debug = args.debug,
        n_jobs = args.n_jobs,
        seed=i
    )
    alg = PolicyGradientBpo(**alg_parameters)
    

    print(text2art(f"==  PG TEST on {args.env} =="))
    print(text2art(f"Trial {i}"))
    print(args)
    print(text2art("Learn Start"))
    alg.learn()
    end_time = time.time()
    alg.save_results()
    # save the policy parameters for each trail
    np.save(f"{dir_name}/policy_params", alg.best_theta)
    print(alg.performance_idx)
    
    time_trial = end_time - start_time
    total_time[i] = time_trial

time_res = { "time": np.array(total_time, dtype=float).tolist()}
with io.open(time_dir, 'w', encoding='utf-8') as f:
        f.write(json.dumps(time_res, ensure_ascii=False, indent=4))
        f.close()

if args.animate:
    eval_env = env_class(horizon=args.horizon, gamma=args.gamma, render_mode="rgb_array")
    frames = evaluate_planning(eval_env, pol, num_episodes=1, horizon=args.horizon, dir=dir_name)
    imageio.mimsave(dir_render, frames, duration=33)