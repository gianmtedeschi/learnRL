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
import pickle
import gym

import json
import io
from tqdm import tqdm
import copy
from adam.adam import Adam

import os
import time



with open(f"/Users/Admin/OneDrive/Documenti/GitHub/learnRL/before_split_policy_1_.pkl", "rb") as g:
                deserialized_policy = pickle.load(g)

leaves= deserialized_policy.history.get_all_leaves()
print(leaves[0].val[0])

#print(deserialized_policy.history.get_current_policy())
        


