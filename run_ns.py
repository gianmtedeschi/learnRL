# Libraries
import argparse
import datetime
from algorithms import PolicyGradientBpo, PolicyGradient, PolicyGradientBpoMIS
from data_processors import IdentityDataProcessor, KernelDataProcessor, RBFMountainCarDataProcessor, NormalizationDataProcessor, RBFMountainCar_v5DataProcessor
from envs import *
from policies import *
from art import *
import pickle 
from common.utils import *
import random
import time
import json
import io
import torch

# Set PyTorch to be deterministic
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)

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
             "reacher", "pendulum", "dam", "ns_cartpole", "cartpole_friction", "mountain_car_simmetric", "mountain_car_4", "mountain_car_5"]
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
parser.add_argument(
    "--kl",
    help="Kl coefficient for behavioural optimization.",
    type=float,
    default=0.0
)
parser.add_argument(
    "--behavioural_std",
    help="The exploration amount of behavioural policy.",
    type=float,
    default=0.1
)
parser.add_argument(
    "--defensive_batchsize",
    help="The number of trajectories to sample from target policy",
    type=int,
    default=0
)

parser.add_argument(
    "--data_processor",
    help="state transformation to apply before policy step",
    type= str,
    default= 'identity',
    choices= ['identity' , 'rbf', 'rbf_mcar', 'normalization', 'rbf_mcar_5']
)

parser.add_argument(
    "--algorithm",
    help = " learning algorithm, off policy or on policy",
    type = str,
    default= 'off_policy',
    choices= ['on_policy', 'off_policy']
)

parser.add_argument(
    "--layers",
    help = "only for deep policy, number of nureons per layer",
    type = int,
    default = 32
)

parser.add_argument(
    "--lr_opt_behavioural",
    help = "lr for behavioural policy optimization through minimation of crossentropy loss",
    type = float,
    default = 1e-4
)

parser.add_argument(
    "--ite_opt_behavioural",
    help = "max iteerations for behavioural policy optimization through minimation of crossentropy loss",
    type = int,
    default = 200
)
parser.add_argument(
    "--params_dir",
    help="Directory from which load policy parameters.",
    type=str,
    default=""
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
    seed = args.starting_seed + i

    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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
    elif args.env == "mountain_car_simmetric":
        env_class = Continuous_MountainCarSimmEnv_v3
        env = Continuous_MountainCarSimmEnv_v3(horizon = args.horizon, gamma = args.gamma, friction = args.friction)
    elif args.env == "river":
        env_class = RiverSwimContinuous
        env = RiverSwimContinuous(horizon=args.horizon, gamma=args.gamma)
    elif args.env == "cartpole":
        env_class = ContCartPole
        env = ContCartPole(horizon=args.horizon, gamma=args.gamma)
    elif args.env == "pendulum":
        env_class = PendulumEnv
        env = PendulumEnv(horizon=args.horizon, gamma=args.gamma, friction = args.friction)
    elif args.env == "dam":
        env_class = Dam
        env = Dam(horizon=args.horizon, gamma=args.gamma)
    elif args.env == "cartpole_friction":
        env_class = ContCartPoleFriction
        env = ContCartPoleFriction(horizon=args.horizon, gamma=args.gamma, mu_p = args.friction)
    elif args.env == "mountain_car_4":
        env_class =  Continuous_MountainCarSimmEnv_v4
        env = Continuous_MountainCarSimmEnv_v4(horizon = args.horizon, gamma = args.gamma, friction = args.friction)
    elif args.env == "mountain_car_5":
        env_class =  Continuous_MountainCarSimmEnv_v5
        env = Continuous_MountainCarSimmEnv_v5(horizon = args.horizon, gamma = args.gamma, friction = args.friction)

    else:
        raise ValueError(f"Invalid env name.")

    #s_dim = env.state_dim
    a_dim = env.action_dim

    """Data Processor"""

    if args.data_processor == 'identity':
        dp = IdentityDataProcessor()
        s_dim = env.state_dim
    elif args.data_processor == 'normalization':
        dp = NormalizationDataProcessor()
        s_dim = env.state_dim
    elif args.data_processor == 'rbf':
        dp = KernelDataProcessor()
        s_dim = dp.num_states
    elif args.data_processor == 'rbf_mcar':
        dp = RBFMountainCarDataProcessor()
        s_dim = dp.num_states
    elif args.data_processor == 'rbf_mcar_5':
        dp = RBFMountainCar_v5DataProcessor()
        s_dim = dp.num_states

    """Policy"""
    if args.pol == "linear_gaussian":

        if args.algorithm =="off_policy" and (args.env == "mountain_car_5" or args.env == "mountain_car_4" or args.env == "pendulum") and args.data_processor!="identity":

            net = nn.Sequential(
            nn.Linear(s_dim, a_dim, bias=False)
            )

            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

            pol = DeepGaussianPolicy(
                parameters = np.load(args.params_dir),
                #parameters=None,
                input_size=s_dim,
                output_size=a_dim,
                model=copy.deepcopy(net),
                std_dev=args.std,
                #std_decay=5e-2,
                std_decay=0,
                std_min=0.25
            )
            tot_params = pol.tot_params
        else:
            tot_params = s_dim * a_dim
            params = np.load(args.params_dir) if args.params_dir else np.zeros(tot_params)
            pol = GaussianPolicy(
                parameters= params,
                dim_state=s_dim,
                dim_action=a_dim,
                std_dev=args.std,
                std_decay=0,
                std_min=1e-6,
                multi_linear=MULTI_LINEAR
            )


    elif args.pol in ["nn", "deep_gaussian"]:
        net = nn.Sequential(
            nn.Linear(s_dim, args.layers, bias=True),
            nn.Tanh(),
            nn.Linear(args.layers, args.layers, bias=True),
            nn.Tanh(),
            nn.Linear(args.layers, a_dim, bias=False),
            nn.Tanh()
        )

        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        
        # model_desc = dict(
        #     layers_shape=[(s_dim, args.layers), (args.layers, args.layers), (args.layers, a_dim)]
        # )



        # PER PENDOLO PER USARE LA LINEAR MA FARE OPTIMIZTION NON IN FORMA CHIUSA
        # net = nn.Sequential(
        #     nn.Linear(s_dim, a_dim, bias=False),
        # )

        # model_desc = dict(
        #     layers_shape=[(s_dim, a_dim)]
        # )

        if args.pol == "nn":
            pol = NeuralNetworkPolicy(
                parameters=None,
                input_size=s_dim,
                output_size=a_dim,
                model=copy.deepcopy(net)
            )

        elif args.pol == "deep_gaussian":
            pol = DeepGaussianPolicy(
                #parameters= np.load(args.params_dir),
                parameters=None,
                input_size=s_dim,
                output_size=a_dim,
                model=copy.deepcopy(net),
                std_dev=args.std,
                #std_decay=5e-2,
                std_decay=0,
                std_min=0.25
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

    if args.algorithm == "off_policy": 
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
            seed=seed,
            kl_reg = args.kl,
            behavioural_std = args.behavioural_std,
            defensive_batch_size = args.defensive_batchsize,
            max_iter_optimization = args.ite_opt_behavioural,
            lr_behavioral = args.lr_opt_behavioural

        )
        alg = PolicyGradientBpo(**alg_parameters)

    elif args.algorithm == "on_policy":
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
            seed=seed,
            # defensive batchsize default
        )
        alg = PolicyGradient(**alg_parameters)
    

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
    # non c'è l'attrito qua 
    eval_env = env_class(horizon=args.horizon, gamma=args.gamma, render_mode="rgb_array")
    frames = evaluate_planning(eval_env, pol, dp, num_episodes=1, horizon=args.horizon, dir=dir_name)
    imageio.mimsave(dir_render, frames, duration=33)