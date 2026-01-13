# Libraries
import argparse
import datetime
from algorithms import PolicyGradientSplit, PolicyGradient, ParameterPolicyGradientSplit
from data_processors import IdentityDataProcessor
from envs import *
from policies import *
from art import *
import pickle 

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--dir",
    help="Directory in which save the results.",
    type=str,
    default="./results/"
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
    default="agaps",
    choices=["pg", "agaps", "pgaps"]
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
    choices=["gaussian", "split_gaussian", "deep_gaussian"]
)
parser.add_argument(
    "--env",
    help="The environment.",
    type=str,
    default="swimmer",
    choices=["swimmer", "half_cheetah", "ant", "lq", "minigolf", "mountain_car"]
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
    "--n_jobs",
    help="Number of parallel jobs.",
    type=int,
    default=-1
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
    default=1
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
parser.add_argument(
    "--max_splits",
    help="Maximum number of division.",
    type=int,
    default=30,
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




args = parser.parse_args()

if args.std < 1:
    string_var = str(args.std).replace(".", "")
else:
    string_var = str(int(args.std))

# Build
base_dir = args.dir
base_dir += "_" + datetime.datetime.now().strftime("%m_%d-%H_%M_")

for i in range(args.n_trials):
    torch.manual_seed(i)
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
    elif args.env == "lq":
        env_class = LQ
        env = LQ(horizon=args.horizon, gamma=args.gamma, action_dim=args.lq_action_dim, state_dim=args.lq_state_dim)
        MULTI_LINEAR = bool(args.lq_action_dim > 1)
    elif args.env == "minigolf":
        env_class = MiniGolf
        env = MiniGolf(horizon=args.horizon, gamma=args.gamma)
    elif args.env == "mountain_car":
        env_class = Continuous_MountainCarEnv
        env = Continuous_MountainCarEnv(horizon=args.horizon, gamma=args.gamma)
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
            parameters=np.zeros(tot_params),
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
            parameters=np.zeros(tot_params),
            dim_state=s_dim,
            dim_action=a_dim,
            std_dev=args.std,
            std_decay=0,
            std_min=1e-6,
            deterministic=args.deterministic,
            linear=args.linear,
            max_splits=args.max_splits
        )
    elif args.pol in ["nn", "deep_gaussian"]:
        net = nn.Sequential(
            nn.Linear(s_dim, 100, bias=False),
            nn.Tanh(),
            nn.Linear(100, 50, bias=False),
            nn.Tanh(),
            nn.Linear(50, 25, bias=False),
            nn.Tanh(),
            nn.Linear(25, a_dim, bias=False),
            nn.Tanh()
        )
        model_desc = dict(
            layers_shape=[(s_dim, 100), (100, 50), (50, 25), (25, a_dim)]
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
        else:
            raise ValueError("Invalid nn policy name.")
        tot_params = pol.tot_params
    else:
        raise ValueError(f"Invalid policy name.")

    
    dir_name += f"{tot_params}_std_{string_var}"
    dir_name += f"_alpha_{str(args.alpha).replace('.', '')}"
    if args.linear:
        dir_name += f"_linear"
    dir_name = base_dir + dir_name + "/" + f"trial_{i}"

    """Algorithms"""
    if args.alg == "pg":
        alg_parameters = dict(
            lr=[args.lr],
            lr_strategy=args.lr_strategy,
            estimator_type=args.estimator,
            # initial_theta=[0] * tot_params,
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
            n_jobs=args.n_jobs
        )
        alg = PolicyGradient(**alg_parameters)
    elif args.alg == "agaps":
        alg_parameters = dict(
            lr=[args.lr],
            lr_strategy=args.lr_strategy,
            estimator_type=args.estimator,
            # initial_theta=[0] * tot_params,
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
            alpha=args.alpha,
            max_splits=args.max_splits,
            n_jobs=args.n_jobs
        )
        alg = PolicyGradientSplit(**alg_parameters)
    elif args.alg == "pgaps":
        alg_parameters = dict(
            lr=[args.lr],
            lr_strategy=args.lr_strategy,
            estimator_type=args.estimator,
            initial_rho=[0] * tot_params,
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
        )
        alg = ParameterPolicyGradientSplit(**alg_parameters)
    else:
        raise ValueError("Invalid algorithm name.")
    

    print(text2art(f"== {args.alg} TEST on {args.env} =="))
    print(text2art(f"Trial {i}"))
    print(args)
    print(text2art("Learn Start"))
    alg.learn()
    alg.save_results()
    print(alg.performance_idx)