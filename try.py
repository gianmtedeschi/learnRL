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

import json
import io
from tqdm import tqdm
import copy
from adam.adam import Adam

import os
import time

low = np.array(
            [0, 5], dtype=np.float32
        )
high = np.array(
            [-1, 1], dtype=np.float32
        )


v=np.array(np.random.uniform(low=low, high=high))
v= np.array(v)
v1= np.array(v[0])
print(v1.shape)