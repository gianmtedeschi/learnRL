"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
from typing import Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
import envs # for registration
import logging
import math
from gymnasium import spaces
from gymnasium.utils import seeding

logger = logging.getLogger(__name__)

class NsCartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):


    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self, sutton_barto_reward: bool = False, 
        render_mode: Optional[str] = None,
        external_force_probability: float = 0.0,  # mixture parameter
        # Gaussian distribution parameters:
        external_force_mean: float = .0, # mean value of the external force
        external_force_std: float = .0, # std deviation, namely "maximum" force magnitude under gaussian noise
        # Beta distribution parameters:
        external_force_alpha: float = .3,
        external_force_beta: float = .3,
        external_force_mag : float = 2.0, # Force magnitude when sampling from beta, catastrophic event
        horizon = 500,
        gamma = 1
    ):
        self.horizon = horizon
        self.gamma = gamma

        self._sutton_barto_reward = sutton_barto_reward

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        # maybe we would need to modify this
        self.theta_threshold_radians = 24 * 2 * math.pi / 360
        self.x_threshold = 4.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians * 2,
                np.inf,
            ],
            dtype=np.float32,
        )

        # external force parameters
        self.external_force_probability = external_force_probability
        # gaussian params init
        self.external_force_mean = external_force_mean
        self.external_force_std = external_force_std
        # beta params init
        self.external_force_alpha = external_force_alpha
        self.external_force_beta = external_force_beta
        self.external_force_mag = external_force_mag

        
        # discrete action space
        # self.action_space = spaces.Discrete(2)
        # continuous action space
        self.action_space = spaces.Box(low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=float)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]   

        self.render_mode =  render_mode

        self.screen_width = 1200
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state: np.ndarray | None = None

        self.steps_beyond_terminated = None

    def step(self, action):
        # check for discrete action spaces
        # assert self.action_space.contains(
        #     action
        # ), f"{action!r} ({type(action)}) invalid"
        
        # check valid action continuous
        action = np.ravel(action)
        assert self.action_space.contains(action), type(action)

        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
    

        # input saturation
        force = np.clip(action, -self.force_mag, self.force_mag)
        
        # discrete action space 
        # force = self.force_mag if action == 1 else -self.force_mag
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # FRICTION
        
        # Increasing friction with theta
        mu_p = 0.01 + 0.09 * (1 - np.exp(-((theta / self.theta_threshold_radians)*2, 20))) / (1 - np.exp(-20))
        
        # No friction
        # mu_p = 0.01 # const friction, minimum
        
        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * np.square(theta_dot) * sintheta
        ) / self.total_mass
        
        # Stochastic External Tip Force Injection
        if bool(theta < self.theta_threshold_radians or theta > (2 * np.pi - self.theta_threshold_radians)):
            if self.np_random.random() >= self.external_force_probability:
                # sample disturbance from a gaussian distribution 
                F_tip = self.np_random.normal(self.external_force_mean, self.external_force_std)
            else:
                # sample disturbance from the Beta
                print("disturbacnce!")
                """
                TO-DO: signal when the beta is sampled during evaluation for feedback.
                - render the environment only when the beta is sampled to check behaviour of the agent.
                """
                beta_sample = self.np_random.beta(self.external_force_alpha,self.external_force_beta)
                external_force_min = -self.external_force_mag
                external_force_max = self.external_force_mag
                F_tip = external_force_min+beta_sample*(external_force_max-external_force_min)
            
            # compute modified angular acceleration
            extra_term = (2*F_tip)/self.masspole
            thetaacc = (self.gravity * sintheta - costheta * temp - (mu_p*theta_dot/self.polemass_length)+extra_term) / (
                self.length
                * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
            )
        else:
            thetaacc = (self.gravity * sintheta - costheta * temp - (mu_p*theta_dot/self.polemass_length)) / (
                self.length
                * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
                )
        
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        # Wrap theta to [0, 2*pi]
        theta = theta % (2 * np.pi)
        
        # update the state after integration
        self.state = np.array((x, x_dot.item(), theta, theta_dot.item()), dtype=np.float64)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
        )
        termination_penalty = 0.0
        if terminated:
            termination_penalty = 100.0  # or some value tuned via experiments

        # the reward for the environment - negative to encode notion of cost
        reward = np.cos(theta.item())
        reward -=  0.001 * (theta_dot.item()**2) + 0.001*(x_dot.item()**2)+0.001*(x.item()**2)
        reward -= termination_penalty # extra penalty for going out of bounds
        # add termination penalty if goes out of bounds
        # if self.render_mode == "human":
        #     self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, terminated, False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed) # reset the random seed of the environment
        
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        if options is not None and "x0" in options:
            assert (
                isinstance(options["x0"], np.ndarray)
                and options["x0"].shape == (4,)
            ), "Invalid state object"
            self.state = options["x0"]
            self.steps_beyond_terminated = None # reset the terminated boolean

        else: # default init on the vertical position
            # self.render_mode = None
            # low, high = utils.maybe_parse_reset_bounds(
            #     options, -0.0001, 0.0001  # default low
            # )  # default high
            # self.state = self.np_random.uniform(low=low, high=high, size=(4,))
            # self.steps_beyond_terminated = None

            x = 0.0
            x_dot = 0.0
            theta = 0.0 + self.np_random.uniform(-0.001, 0.001)  # near vertical
            theta_dot = 0.0
            self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
            self.steps_beyond_terminated = None

        # if self.render_mode == "human":
        #    self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False