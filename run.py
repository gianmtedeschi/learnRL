# Libraries
import argparse
import datetime
from algorithms import PolicyGradientSplit, PolicyGradient
from data_processors import IdentityDataProcessor
from envs import *
from policies import *
from art import *

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
    choices=["pg", "split"]
)
parser.add_argument(
    "--estimator",
    help="The algorithm to use.",
    type=str,
    default="GPOMDP",
    choices=["REINFORCE", "GPOMDP"]
)
parser.add_argument(
    "--std",
    help="The exploration amount.",
    type=float,
    default=0.1
)
parser.add_argument(
    "--pol",
    help="The policy used.",
    type=str,
    default="split_gaussian",
    choices=["gaussian", "split_gaussian"]
)
parser.add_argument(
    "--env",
    help="The environment.",
    type=str,
    default="swimmer",
    choices=["swimmer", "half_cheetah", "hopper", "ant", "lq", "pendulum", "mountain_car", "minigolf", "pusher", "reacher"]
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
    help="The baseline choosen.",
    type=float,
    default=1e-1,
)



args = parser.parse_args()

if args.std < 1:
    string_var = str(args.std).replace(".", "")
else:
    string_var = str(int(args.std))

# Build
base_dir = args.dir
base_dir += "_" + datetime.datetime.now().strftime("%m_%d-%H_%M_")

for i in range(args.n_trials):
    np.random.seed(i)
    dir_name = f"{args.alg}_{args.ite}_{args.env}_{args.horizon}_{args.lr_strategy}_"
    dir_name += f"{str(args.lr).replace('.', '')}_{args.pol}_batch_{args.batch}_"
    if args.clip:
        dir_name += "clip_"
    else:
        dir_name += "noclip_"

    """Environment"""
    MULTI_LINEAR = False

    if args.env == "swimmer":
        env_class = Swimmer
        env = Swimmer(horizon=args.horizon, gamma=args.gamma, render=False, clip=bool(args.clip))
        MULTI_LINEAR = True
    elif args.env == "half_cheetah":
        env_class = HalfCheetah
        env = HalfCheetah(horizon=args.horizon, gamma=args.gamma, render=False, clip=bool(args.clip))
        MULTI_LINEAR = True
    elif args.env == "ant":
        env_class = Ant
        env = Ant(horizon=args.horizon, gamma=args.gamma, render=False, clip=bool(args.clip))
        MULTI_LINEAR = True
    elif args.env == "hopper":
        env_class = Hopper
        env = Hopper(horizon=args.horizon, gamma=args.gamma, render=False, clip=bool(args.clip))
        MULTI_LINEAR = True
    elif args.env == "lq":
        env_class = LQ
        env = LQ(horizon=args.horizon, gamma=args.gamma, action_dim=args.lq_action_dim, state_dim=args.lq_state_dim)
        MULTI_LINEAR = bool(args.lq_action_dim > 1)
    elif args.env == "pendulum":
        env_class = PendulumEnv
        env = PendulumEnv(horizon=args.horizon, gamma=args.gamma)
    elif args.env == "mountain_car":
        env_class = Continuous_MountainCarEnv
        env = Continuous_MountainCarEnv(horizon=args.horizon, gamma=args.gamma)
    elif args.env == "minigolf":
        env_class = MiniGolf
        env = MiniGolf(horizon=args.horizon, gamma=args.gamma)
    elif args.env == "pusher":
        env_class = Pusher
        env = Pusher(horizon=args.horizon, gamma=args.gamma, clip=bool(args.clip))
        MULTI_LINEAR = True
    elif args.env == "reacher":
        env_class = Reacher
        env = Reacher(horizon=args.horizon, gamma=args.gamma, clip=bool(args.clip))
        MULTI_LINEAR = True
    else:
        raise ValueError(f"Invalid env name.")

    s_dim = env.state_dim
    a_dim = env.action_dim

    """Data Processor"""
    dp = IdentityDataProcessor()

    """Policy"""
    if args.pol == "gaussian":
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
    elif args.pol == "split_gaussian":
        tot_params = a_dim
        pol = SplitGaussianPolicy(
            parameters=np.ones(tot_params),
            dim_state=s_dim,
            dim_action=a_dim,
            std_dev=args.std,
            std_decay=0,
            std_min=1e-6,
        )
    else:
        raise NotImplementedError
    
    dir_name += f"{tot_params}_std_{string_var}"
    dir_name += f"_alpha_{str(args.alpha).replace('.', '')}"
    dir_name = base_dir + dir_name + "/" + f"trial_{i}"

    """Algorithms"""
    if args.alg == "pg":
        alg_parameters = dict(
            lr=[args.lr],
            lr_strategy=args.lr_strategy,
            estimator_type=args.estimator,
            initial_theta=[0] * tot_params,
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
        )
        alg = PolicyGradient(**alg_parameters)
    elif args.alg == "split":
        alg_parameters = dict(
            lr=[args.lr],
            lr_strategy=args.lr_strategy,
            estimator_type=args.estimator,
            initial_theta=[0] * tot_params,
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
            split_grid=None,
            alpha=args.alpha
        )
        alg = PolicyGradientSplit(**alg_parameters)
    else:
        raise ValueError("Invalid algorithm name.")
    

    print(text2art(f"== {args.alg} TEST on {args.env} =="))
    print(text2art(f"Trial {i}"))
    print(args)
    print(text2art("Learn Start"))
    alg.learn()
    alg.save_results()
    print(alg.performance_idx)