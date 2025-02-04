import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class RiverSwimContinuous(gym.Env):

    def __init__(self, dim=3, gamma=0.998, small=5, large=50, horizon=30):

        self.horizon = horizon
        self.small = small
        self.large = large
        self.gamma = gamma
        self.dim = dim

        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = 0
        self.max_position = dim

        self.viewer = None

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.min_position, high=self.max_position,
                                            shape=(1,), dtype=np.float32)

        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        self.reset()

    def _get_prob(self, action):

        prob = np.zeros(3)
        if action <= 0:
            r = (self.action_space.low - action) / self.action_space.low
            prob[0] = 0.9 * r # 1 - r * 0.9
            prob[1] = 1 - prob[0]
        else:
            r = action / self.action_space.high
            prob[0] = 0.1
            # prob[1] = 0.9 - r * 0.3
            # prob[2] = 1 - prob[1] - prob[0]
            prob[2] = 0.3 * r
            prob[1] = 1 - prob[0] - prob[2]

        return prob

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        prob = self._get_prob(action)
        dir = self.np_random.choice(3, p=prob)

        pos = self.state

        if dir == 0:
            new_state = pos - np.abs(action)
        elif dir == 1:
            new_state = pos
        else:
            new_state = pos + np.abs(action)

        reward = 0.
        if action <= 0 and pos <= 1:
            reward = self.small
        elif action >= 0 and pos >= self.dim - 1:
            reward = self.large

        self.state = np.clip(new_state, self.observation_space.low, self.observation_space.high)


        return self.state, reward, False, {}

    def reset(self, seed=None):
        self.state = 0


# env = RiverSwimContinuous()

# T = 200
# for act in [-1, 0, 1]:
#     All = []
#     for _ in range(1000):
#         s = env.reset()
#         rr = 0
#         for t in range(T):
#             a = act + np.random.randn() * 0.5
#             s, r, _, _ = env.step(a)
#             rr += r * env.gamma ** t
#         All.append(rr)

#     print('action = ' + str(act), np.mean(All))