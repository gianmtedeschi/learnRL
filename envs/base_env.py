"""Base environment class"""

# imports
from abc import ABC, abstractmethod
import copy
import numpy as np


# class
class BaseEnv(ABC):
    def __init__(
            self,
            horizon: int = 0,
            gamma: float = 0.99,
            verbose: bool = False,
            clip: bool = True
    ) -> None:
        self.horizon = horizon

        assert 0 <= gamma <= 1, "[ERROR] Invalid Discount Factor value."
        self.gamma = gamma

        self.time = 0
        self.verbose = verbose
        self.state_dim = 0
        self.action_dim = 0
        self.clip = clip

    @abstractmethod
    def step(self, action):
        """todo"""
        pass

    def reset(self, seed=None) -> None:
        """todo"""
        self.time = 0



class MujocoBase(BaseEnv, ABC):
    def __init__(
            self,
            horizon: int = 0,
            gamma: float = 0.99,
            verbose: bool = False,
            clip: bool = True,
            seed: int = None
    ) -> None:
        super().__init__(horizon=horizon, gamma=gamma, verbose=verbose, clip=clip)
        self.gym_env = None
        self.state = None
        self.seed = seed

    def step(self, action):
        obs, reward, done, info, _ = self.gym_env.step(action)
        self.state = copy.deepcopy(obs)
        return obs, reward, done, info

    def reset(self, seed=None):
        obs = self.gym_env.reset(seed=seed)
        self.state = copy.deepcopy(obs[0])
        return obs

    def render(self, mode='human'):
        return self.gym_env.render()

    def close(self):
        self.gym_env.close()

    def sample_action(self):
        return self.gym_env.action_space.sample()

    def sample_state(self, args: dict = None):
        return self.gym_env.observation_space.sample()
    
    def set_state(self, state):
        pass