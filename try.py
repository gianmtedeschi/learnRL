import numpy as np
import scipy

import scipy.stats as stats
import gym
from gym import spaces
from gym.utils import seeding
delta= 0.1

state = np.array(seeding.np_random(1)[0].uniform(low=-5.,high=5.,size=2))

print(state)