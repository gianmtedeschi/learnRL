"""
classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
# Libraries
import logging
import math
import gym
import gym.spaces as spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)


class ContCartPole(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, horizon, gamma) -> None:
        # todo do better
        self.horizon = horizon
        self.gamma = gamma
        self.action_bounds = [-10, 10]
        self.state_dim = 4
        self.action_dim = 1

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5   # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within
        # bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float64).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float64).max])

        self.action_space = spaces.Box(
            low=-self.force_mag, high=self.force_mag,shape=(1,),
            dtype=np.float64
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float64)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        self.track = None
        self.axle = None
        self.poletrans = None
        self.carttrans = None
        self.np_random = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.ravel(action)
        action = np.clip(
            action,
            0, 1
        )
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = np.clip(action, -self.force_mag, self.force_mag)
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array([x, x_dot[0], theta, theta_dot[0]])
        done = (x < -self.x_threshold or x > self.x_threshold
                or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians)
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already "
                               "returned done = True. You should always call 'reset()' once you "
                               "receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        return np.ravel(self.state), reward, done, {}

    def reset(self, initial=None):
        if initial is None:
            self.state = np.array(self.np_random.uniform(low=-0.05, high=0.05, size=(4,)))
        else:
            self.state = initial
        self.steps_beyond_done = None
        return np.ravel(self.state)

    def render(self, mode='human', close=False):
        pass