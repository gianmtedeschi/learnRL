
import numpy as np
import gym
import numpy as np
from gym import spaces

"""
Cyclostationary Dam Control
Info
----
  - State space: 2D Box (storage,day)
  - Action space: 1D Box (release decision)
  - Parameters: capacity, demand, flooding threshold, inflow mean per day, inflow std, demand weight, flooding weigt=ht
References
----------
  - Simone Parisi, Matteo Pirotta, Nicola Smacchia,
    Luca Bascetta, Marcello Restelli,
    Policy gradient approaches for multi-objective sequential decision making
    2014 International Joint Conference on Neural Networks (IJCNN)
  - A. Castelletti, S. Galelli, M. Restelli, R. Soncini-Sessa
    Tree-based reinforcement learning for optimal water reservoir operation
    Water Resources Research 46.9 (2010)
  - Andrea Tirinzoni, Andrea Sessa, Matteo Pirotta, Marcello Restelli.
    Importance Weighted Transfer of Samples in Reinforcement Learning.
    International Conference on Machine Learning. 2018.
"""


class Dam(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, horizon=360, gamma=1, alpha=0.5, beta=0.5, penalty_on=False):
        self.horizon = horizon
        self.gamma = gamma

        self.DEMAND = 10.0  # Water demand -> At least DEMAND/day must be supplied or a cost is incurred
        self.FLOODING = 300.0  # Flooding threshold -> No more than FLOODING can be stored or a cost is incurred
        self.MIN_STORAGE = 50.0  # Minimum storage capacity -> At most max{S - MIN_STORAGE, 0} must be released
        self.MAX_STORAGE = 500.0  # Maximum storage capacity -> At least max{S - MAX_STORAGE, 0} must be released

        self.INFLOW_MEAN = self._get_inflow_profile()
        self.INFLOW_STD = 2.0  # Random inflow std

        assert alpha + beta == 1.0  # Check correctness
        self.ALPHA = alpha  # Weight for the flooding cost
        self.BETA = beta  # Weight for the demand cost

        self.penalty_on = penalty_on  # Whether to penalize illegal actions or not

        self.storage = None
        self.day = None

        # Gym attributes
        self.viewer = None

        self.action_space = spaces.Discrete(21)
        self.observation_space = spaces.Box(low=-5 * np.ones(7),
                                            high=5 * np.ones(7),
                                            dtype=np.float32,
                                            shape=(7,))

        self.state_dim = self.observation_space.shape[0]
        self.action_dim = 1
        
        # Initialization
        # self.seed()
        self.reset()

    # @abstractmethod
    def _get_inflow_profile(self):
        y = np.zeros(360)
        x = np.arange(360)
        y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359) + 0.5
        y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359) / 2 + 0.5
        y[240:] = np.sin(x[240:] * 3 * np.pi / 359) + 0.5
        return y * 8 + 4
    
    def day_step(self, action, day, storage):
        # action = np.clip(action,
        #         0,
        #         1,
        #         dtype=np.float64
        #     )
        action = 0 + (action - (-1)) * (20 - (-1)) / (1 - (-1))
        
        # action = float(action)

        # Bound the action
        actionLB = max(storage - self.MAX_STORAGE, 0.0)
        actionUB = max(storage - self.MIN_STORAGE, 0.0)

        # Penalty proportional to the violation
        bounded_action = min(max(action, actionLB), actionUB)
        penalty = -abs(bounded_action - action) * self.penalty_on

        # Transition dynamics
        action = bounded_action
        inflow = self.INFLOW_MEAN[int(day - 1)] + np.random.randn() * self.INFLOW_STD
        nextstorage = max(storage + inflow - action, 0.0)

        # Cost due to the excess level wrt the flooding threshold
        reward_flooding = -max(storage - self.FLOODING, 0.0) / 4

        # Deficit in the water supply wrt the water demand
        reward_demand = -max(self.DEMAND - action, 0.0) ** 2

        # The final reward is a weighted average of the two costs
        reward = self.ALPHA * reward_flooding + self.BETA * reward_demand + penalty

        # Get next day
        nextday = day + 1 if day < 360 else 1

        return reward, nextday, nextstorage

    def step(self, action):
        action = float(action)

        # Get current state
        reward = 0
        # for _ in range(5):
        day = self.day
        storage = self.storage
        curr_reward, nextday, nextstorage = self.day_step(action, day, storage)
        reward += curr_reward
        self.storage = nextstorage
        self.day = nextday

        reward *= 0.01
        return self.get_state(), reward, False, {}

    def get_state(self):
        # Original state version
        # return np.array([self.storage, self.day])

        # Sin basis version of the day
        # norm_storage = 2 * (self.storage - self.MIN_STORAGE) / (self.MAX_STORAGE - self.MIN_STORAGE) - 1
        # return np.array([norm_storage,
        #                 np.sin(2 * np.pi / 359 * self.day),
        #                 np.sin(3 * np.pi / 359 * self.day),
        #                 np.sin(4 * np.pi / 359 * self.day)
        #                 ])

        # Distance from "cardinal" time points
        c_list = np.array([60, 120, 180, 240, 300, 360])
        d_list = np.abs(c_list - self.day)
        norm_d = 2 * (d_list - 0) / (360 - 0) - 1
        norm_d = norm_d.tolist()
        norm_storage = 2 * (self.storage - self.MIN_STORAGE) / (self.MAX_STORAGE - self.MIN_STORAGE) - 1

        self.state = np.array([norm_storage] + norm_d, dtype=np.float32)
        return np.array([norm_storage] + norm_d, dtype=np.float32)

    def reset(self, seed=None):
        # init_days = np.array([1, 1, 1])
        # self.state = [np.random.uniform(self.MIN_STORAGE, self.MAX_STORAGE),
        #              init_days[np.random.randint(low=0, high=3)]]
        self.storage = 200
        self.day = 1

        return self.get_state()


# class DamInflow1(Dam):

#     def _get_inflow_profile(self):
#         y = np.zeros(360)
#         x = np.arange(360)
#         y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359) + 0.5
#         y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359) / 2 + 0.5
#         y[240:] = np.sin(x[240:] * 3 * np.pi / 359) + 0.5
#         return y * 8 + 4


# class DamInflow2(Dam):

#     def _get_inflow_profile(self):
#         y = np.zeros(360)
#         x = np.arange(360)
#         y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359) / 2 + 0.25
#         y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359 + np.pi) * 3 + 0.25
#         y[240:] = np.sin(x[240:] * 3 * np.pi / 359 + np.pi) / 4 + 0.25
#         return y * 8 + 4


# class DamInflow3(Dam):

#     def _get_inflow_profile(self):
#         y = np.zeros(360)
#         x = np.arange(360)
#         y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359) * 3 + 0.25
#         y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359) / 4 + 0.25
#         y[240:] = np.sin(x[240:] * 3 * np.pi / 359) / 2 + 0.25
#         return y * 8 + 4


# class DamInflow4(Dam):

#     def _get_inflow_profile(self):
#         y = np.zeros(360)
#         x = np.arange(360)
#         y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359) + 0.5
#         y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359) / 2.5 + 0.5
#         y[240:] = np.sin(x[240:] * 3 * np.pi / 359) + 0.5
#         return y * 7 + 4


# class DamInflow5(Dam):

#     def _get_inflow_profile(self):
#         y = np.zeros(360)
#         x = np.arange(360)
#         y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359 - np.pi / 12) / 2 + 0.5
#         y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359 - np.pi / 12) / 2 + 0.5
#         y[240:] = np.sin(x[240:] * 3 * np.pi / 359 - np.pi / 12) / 2 + 0.5
#         return y * 8 + 5


# class DamInflow6(Dam):

#     def _get_inflow_profile(self):
#         y = np.zeros(360)
#         x = np.arange(360)
#         y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359 + np.pi / 8) / 3 + 0.5
#         y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359 + np.pi / 8) / 3 + 0.5
#         y[240:] = np.sin(x[240:] * 3 * np.pi / 359 + np.pi / 8) / 3 + 0.5
#         return y * 8 + 4


# class DamInflow7(Dam):
#     def _get_inflow_profile(self):
#         y = np.zeros(360)
#         x = np.arange(360)
#         y[0:120] = np.sin(x[0:120] * 3 * np.pi / 359) + 0.5
#         y[120:240] = np.sin(x[120:240] * 3 * np.pi / 359) / 3 + 0.5
#         y[240:] = np.sin(x[240:] * 3 * np.pi / 359) * 2 + 0.5
#         return y * 8 + 5