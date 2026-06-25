"""Linear Quadratic Regulator env.

s_{t+1} = A s_t + B a_t + noise
r_{t+1} = - ( s_t^T Q s_t + a_t^T R a_t + 2 a_t^T M_mix s_t )
(an optional terminal cost s_H^T Q_final s_H is added at the horizon)

This is a merge of the original repo `LQ` with a more general, fully
matrix-form implementation:

  * arbitrary A, B, Q, R (scalar or matrix), optional cross term `M_mix`
    and terminal cost `Q_final`, with state_dim != action_dim supported;
  * a complete closed-form value toolkit (infinite- and finite-horizon
    optimal gains/returns, V/Q functions, quadratic Q-features).

The repo-facing interface is intentionally unchanged: it still subclasses
`BaseEnv`, keeps the old gym API (`step` returns a 4-tuple, `reset` returns
the bare state, `self.state` is settable, `get_state`/`seed`), and the
`horizon`, `gamma`, `state_dim`, `action_dim`, `A`, `B`, `Q`, `R` attributes.
The default system reproduces the legacy one: A = 0.9 I, B = 0.9 I, Q = R = I,
uniform initial state on [-5, 5]^d.

Gain convention: K is `action_dim x state_dim` and the linear policy is
`a = K s` (consistent with `computeOptimalK` and the rest of the repo).
"""

# imports
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
import warnings
from numbers import Number

try:
    from envs.base_env import BaseEnv
except ModuleNotFoundError:  # allow running this file directly: `python envs/lq.py`
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from envs.base_env import BaseEnv


# class
class LQ(BaseEnv):
    """Gym environment implementing an LQR problem."""

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self,
                 horizon=50,
                 gamma=0.98,
                 action_dim=1,
                 state_dim=1,
                 noise=0,
                 max_action=10.0,
                 seed=None,
                 # --- optional generalisations (defaults reproduce legacy LQ) ---
                 A=None,
                 B=None,
                 Q=None,
                 R=None,
                 M_mix=0.0,
                 Q_final=0.0,
                 init_dist="uniform",
                 init_bound=5.0,
                 init_mean=0.0,
                 init_std=1.0,
                 check_controllability=False) -> None:

        super().__init__(horizon=horizon, gamma=gamma)

        # ---- system matrices --------------------------------------------------
        # "repo mode": dimensions are given, matrices fall back to the legacy
        # defaults (A = 0.9 I, B = 0.9 I, Q = R = I).
        # "matrix mode": A is provided, dimensions are inferred from A and B.
        if A is None:
            self.state_dim = int(state_dim)
            self.action_dim = int(action_dim)
            A = 0.9 * np.eye(self.state_dim)
            B = 0.9 * np.eye(self.state_dim, self.action_dim)
            Q = np.eye(self.state_dim) if Q is None else Q
            R = np.eye(self.action_dim) if R is None else R
        else:
            A = self._as_matrix(A)
            if A.shape[0] != A.shape[1]:
                raise ValueError("A must be a square matrix")
            self.state_dim = A.shape[0]
            if B is None:
                B = 0.9 * np.eye(self.state_dim, self.action_dim)
            B = self._as_matrix(B, rows=self.state_dim)
            self.action_dim = B.shape[1]
            Q = np.eye(self.state_dim) if Q is None else Q
            R = np.eye(self.action_dim) if R is None else R

        ds, da = self.state_dim, self.action_dim
        self.A = self._as_matrix(A, rows=ds, cols=ds)
        self.B = self._as_matrix(B, rows=ds, cols=da)
        self.Q = self._symmetric_matrix(Q, ds, "Q", strict=True)
        self.R = self._symmetric_matrix(R, da, "R", strict=True)
        self.Q_final = self._symmetric_matrix(Q_final, ds, "Q_final", strict=False)

        # cross term: cost contribution 2 a^T M_mix s  (M_mix is da x ds)
        if np.isscalar(M_mix):
            M_mix = np.zeros((da, ds)) if np.isclose(M_mix, 0.0) \
                else M_mix * np.ones((da, ds))
        M_mix = np.asarray(M_mix, dtype=float)
        if M_mix.shape != (da, ds):
            raise ValueError(f"M_mix should be a {da}x{ds} matrix")
        self.M_mix = M_mix

        if check_controllability:
            self._check_controllability()

        # ---- horizon / discount ----------------------------------------------
        self.horizon = horizon
        self.gamma = gamma

        # ---- bounds & noise ---------------------------------------------------
        self.max_pos = 10.0 * np.ones(ds)            # used for obs space / r_max
        self.max_action = max_action * np.ones(da)
        # legacy scalar `noise` -> per-dimension std vector; kept for analytics
        if np.isscalar(noise):
            noise_std = float(noise) * np.ones(ds)
        else:
            noise_std = np.asarray(noise, dtype=float)
        self.noise_std = noise_std
        self.sigma_noise = np.diag(noise_std)        # legacy attribute

        # ---- initial-state distribution --------------------------------------
        # Default: uniform on [-init_bound, init_bound]^ds (legacy behaviour).
        # Also supports a Gaussian init (mean, std). `init_second_moment` is the
        # E[x0 x0^T] used by the closed-form return computations.
        self.init_dist = init_dist
        if init_dist == "uniform":
            self.init_bound = init_bound * np.ones(ds) if np.isscalar(init_bound) \
                else np.asarray(init_bound, dtype=float)
            self.init_second_moment = np.diag((self.init_bound ** 2) / 3.0)
        elif init_dist == "gaussian":
            self.init_mean = init_mean * np.ones(ds) if np.isscalar(init_mean) \
                else np.asarray(init_mean, dtype=float)
            self.init_std = init_std * np.ones(ds) if np.isscalar(init_std) \
                else np.asarray(init_std, dtype=float)
            self.init_second_moment = (np.outer(self.init_mean, self.init_mean)
                                       + np.diag(self.init_std ** 2))
        else:
            raise ValueError("init_dist must be 'uniform' or 'gaussian'")

        # ---- gym spaces -------------------------------------------------------
        self.action_space = spaces.Box(low=-self.max_action,
                                       high=self.max_action,
                                       dtype=np.float64)
        self.observation_space = spaces.Box(low=-self.max_pos,
                                            high=self.max_pos,
                                            dtype=np.float64)

        # ---- initialise -------------------------------------------------------
        self.viewer = None
        self.seed(seed)
        self.reset()

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _as_matrix(M, rows=None, cols=None):
        """Coerce a scalar / vector / matrix into a 2-D array (and validate)."""
        if np.isscalar(M):
            if rows is not None and cols is not None:
                return float(M) * np.eye(rows, cols)
            return float(M) * np.eye(1)
        M = np.asarray(M, dtype=float)
        if M.ndim == 1 and rows is not None and M.shape[0] == rows:
            M = M[:, None]
        if M.ndim != 2:
            raise ValueError("expected a 2-D matrix")
        if rows is not None and M.shape[0] != rows:
            raise ValueError(f"matrix should have {rows} rows, got {M.shape[0]}")
        if cols is not None and M.shape[1] != cols:
            raise ValueError(f"matrix should have {cols} cols, got {M.shape[1]}")
        return M

    @staticmethod
    def _symmetric_matrix(M, dim, name, strict):
        if np.isscalar(M):
            M = float(M) * np.eye(dim)
        M = np.asarray(M, dtype=float)
        if M.shape != (dim, dim):
            raise ValueError(f"{name} should be a {dim}x{dim} matrix")
        if not np.allclose(M, M.T):
            raise ValueError(f"{name} should be symmetric")
        eig = np.linalg.eigvalsh(M)
        if strict and not np.all(eig > 0):
            raise ValueError(f"{name} should be symmetric positive definite")
        if (not strict) and not np.all(eig >= -1e-10):
            raise ValueError(f"{name} should be symmetric positive semi-definite")
        return M

    def _check_controllability(self):
        powers = [self.B]
        for _ in range(self.state_dim - 1):
            powers.append(self.A @ powers[-1])
        C = np.concatenate(powers, axis=1)
        if np.linalg.matrix_rank(C) < self.state_dim:
            warnings.warn("The system is not controllable!", UserWarning)

    def _gain(self, K):
        """Coerce a gain into the (action_dim, state_dim) convention a = K s."""
        return np.asarray(K, dtype=float).reshape(self.action_dim, self.state_dim)

    # ------------------------------------------------------------------- gym API
    def step(self, action, render=False):
        # u = np.clip(np.ravel(np.atleast_1d(action)),
        #             -self.max_action, self.max_action).flatten()

        u = np.clip(np.ravel(np.atleast_1d(action)),
            -np.inf, np.inf).flatten()

        cost = (self.state @ self.Q @ self.state
                + u @ self.R @ u
                + 2.0 * (u @ self.M_mix @ self.state))

        noise = self.sigma_noise @ np.random.randn(self.state_dim)
        xn = self.A @ self.state + self.B @ u + noise

        self.state = np.ravel(xn)
        self.timestep += 1
        done = self.timestep >= self.horizon
        if done and not np.allclose(self.Q_final, 0.0):
            cost = cost + self.state @ self.Q_final @ self.state

        return self.get_state(), -float(cost), done, {'danger': 0}

    def reset(self, state=None, seed=None):
        self.timestep = 0
        if seed is not None:
            self.seed(seed)
        if state is not None:
            self.state = np.array(state, dtype=float)
        elif self.init_dist == "uniform":
            self.state = self.np_random.uniform(low=-self.init_bound,
                                                high=self.init_bound)
        else:  # gaussian
            self.state = (self.init_mean
                          + self.np_random.standard_normal(self.state_dim) * self.init_std)
        return self.get_state()

    def get_state(self):
        return np.array(self.state)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_state_dim(self, state_dim):
        self.state_dim = state_dim

    def set_action_dim(self, action_dim):
        self.action_dim = action_dim

    def computer_r_max(self, episodes=None):
        if episodes is not None:
            return self.max_pos ** 2 * self.Q + 0.3 ** 2 * self.R
        return self.max_pos ** 2 * self.Q + self.max_action ** 2 * self.R

    # ============================================================ value toolkit
    # All methods use the convention a = K s with K of shape (action_dim, state_dim).
    def _closed_loop_P(self, K, discount, max_iterations=100):
        """Riccati P for the *fixed* linear policy a = K s (policy evaluation).

        Solves P = S + discount (A + B K)^T P (A + B K), with
        S = Q + K^T R K + K^T M_mix + M_mix^T K the closed-loop stage-cost matrix.
        """
        K = self._gain(K)
        A_cl = self.A + self.B @ K
        S = self.Q + K.T @ self.R @ K + K.T @ self.M_mix + self.M_mix.T @ K
        P = self.Q.copy()
        for _ in range(max_iterations):
            P_next = S + discount * (A_cl.T @ P @ A_cl)
            if np.allclose(P_next, P):
                return P_next
            P = P_next
        warnings.warn("Computation of closed-loop P did not converge")
        return P

    def _computeP2(self, K, discount=None, max_iterations=100):
        """Backward-compatible alias for the policy-evaluation Riccati."""
        discount = self.gamma if discount is None else discount
        return self._closed_loop_P(K, discount, max_iterations)

    def discounted_P_matrix(self, discount=None, max_iterations=100):
        """Optimal (control) Riccati matrix for the discounted problem."""
        discount = self.gamma if discount is None else discount
        P = self.Q.copy()
        for _ in range(max_iterations):
            inverse = np.linalg.inv(self.R + discount * self.B.T @ P @ self.B)
            M = self.M_mix + discount * self.B.T @ P @ self.A
            P_next = self.Q + discount * (self.A.T @ P @ self.A) - M.T @ inverse @ M
            if np.allclose(P_next, P):
                return P_next
            P = P_next
        warnings.warn("Computation of optimal P did not converge")
        return P

    def discounted_optimal_gain(self, discount=None, max_iterations=100):
        """Optimal discounted gain K* (action_dim x state_dim), a = K* s."""
        discount = self.gamma if discount is None else discount
        P = self.discounted_P_matrix(discount, max_iterations)
        inverse = np.linalg.inv(self.R + discount * self.B.T @ P @ self.B)
        return - inverse @ (self.M_mix + discount * self.B.T @ P @ self.A)

    def computeOptimalK(self, discount=None):
        """Optimal linear controller (a = K s). Backward-compatible name."""
        return self.discounted_optimal_gain(discount)

    def _noise_terms(self, P, discount, policy_std):
        """Constant additive return terms from state noise and policy noise."""
        state_term = discount * np.trace(np.diag(self.noise_std ** 2) @ P) / (1.0 - discount)
        action_term = np.trace((self.R + discount * self.B.T @ P @ self.B)
                               * (policy_std ** 2)) / (1.0 - discount)
        return state_term + action_term

    def discounted_optimal_return(self, discount=None, policy_std=0., max_iterations=100):
        """Closed-form optimal discounted return (scalar policy std)."""
        discount = self.gamma if discount is None else discount
        P = self.discounted_P_matrix(discount, max_iterations)
        init_term = np.trace(P @ self.init_second_moment)
        return - init_term - self._noise_terms(P, discount, policy_std)

    def computeJ(self, K, Sigma=1., n_random_x0=None, discount=None, max_iterations=100):
        """Discounted return of the linear policy a = K s + N(0, Sigma).

        Closed form (matrix, multi-dimensional). `n_random_x0` is accepted for
        backward compatibility and ignored. Sigma may be a scalar or a matrix.
        """
        discount = self.gamma if discount is None else discount
        P = self._closed_loop_P(K, discount, max_iterations)
        if np.isscalar(Sigma):
            Sigma = float(Sigma) * np.eye(self.action_dim)
        Sigma = np.asarray(Sigma, dtype=float)

        init_term = np.trace(P @ self.init_second_moment)
        state_term = discount * np.trace(np.diag(self.noise_std ** 2) @ P) / (1.0 - discount)
        action_term = np.trace(Sigma @ (self.R + discount * self.B.T @ P @ self.B)) / (1.0 - discount)
        return - init_term - state_term - action_term

    def discounted_v(self, state, policy_param, discount=None, policy_std=0., max_iterations=100):
        """State-value V(s) of the linear policy a = K s (+ noise)."""
        discount = self.gamma if discount is None else discount
        if not np.allclose(self.M_mix, 0.):
            raise NotImplementedError("discounted_v assumes M_mix = 0")
        P = self._closed_loop_P(policy_param, discount, max_iterations)
        state = np.ravel(state)
        return - state @ P @ state - self._noise_terms(P, discount, policy_std)

    def discounted_q(self, state, action, policy_param, discount=None, policy_std=0., max_iterations=100):
        """Action-value Q(s, a) of the linear policy a = K s (+ noise)."""
        discount = self.gamma if discount is None else discount
        if not np.allclose(self.M_mix, 0.):
            raise NotImplementedError("discounted_q assumes M_mix = 0")
        P = self._closed_loop_P(policy_param, discount, max_iterations)
        state, action = np.ravel(state), np.ravel(action)
        Q_11 = self.Q + discount * self.A.T @ P @ self.A
        Q_12 = discount * self.A.T @ P @ self.B
        Q_21 = discount * self.B.T @ P @ self.A
        Q_22 = self.R + discount * self.B.T @ P @ self.B
        sa_term = - (state @ Q_11 @ state
                     + state @ Q_12 @ action
                     + action @ Q_21 @ state
                     + action @ Q_22 @ action)
        # constant noise terms (discounted, matching the original implementation)
        state_term = discount * np.trace(np.diag(self.noise_std ** 2) @ P) / (1.0 - discount)
        action_term = discount * np.trace((self.R + discount * self.B.T @ P @ self.B)
                                          * (policy_std ** 2)) / (1.0 - discount)
        return sa_term - state_term - action_term

    def q_representation(self, state, action):
        """Quadratic feature vector phi(s, a) for a linear Q-function model."""
        state, action = np.ravel(state), np.ravel(action)
        if state.shape != (self.state_dim,) or action.shape != (self.action_dim,):
            raise ValueError("Invalid state or action shape")
        x = np.concatenate((state, action))
        outer = np.outer(x, x)
        triu = outer[np.triu_indices(self.state_dim + self.action_dim)]
        return np.concatenate((np.ones(1), triu))

    # ---------------------------------------------------- finite-horizon optimum
    def P_matrices(self, horizon, discount=1.):
        """Time-varying optimal Riccati matrices [P_0, ..., P_H = Q_final]."""
        Ps = [self.Q_final]
        for _ in range(horizon):
            P_next = Ps[0]
            M = self.M_mix + discount * self.B.T @ P_next @ self.A
            inverse = np.linalg.inv(self.R + discount * self.B.T @ P_next @ self.B)
            P = self.Q + discount * (self.A.T @ P_next @ self.A) - M.T @ inverse @ M
            Ps.insert(0, P)
        return Ps

    def optimal_gains(self, horizon, discount=1.):
        """Time-varying optimal gains [K_0, ..., K_{H-1}], a_t = K_t s_t."""
        Ps = self.P_matrices(horizon, discount)
        return [- np.linalg.inv(self.R + discount * self.B.T @ Ps[h + 1] @ self.B)
                @ (self.M_mix + discount * self.B.T @ Ps[h + 1] @ self.A)
                for h in range(horizon)]

    def optimal_return(self, horizon, policy_std=0., discount=1.):
        """Closed-form optimal finite-horizon return (scalar policy std)."""
        Ps = self.P_matrices(horizon, discount)
        init_term = np.trace(Ps[0] @ self.init_second_moment)
        state_term = sum(np.trace(discount * np.diag(self.noise_std ** 2) @ Ps[h + 1])
                         for h in range(horizon))
        action_term = sum(np.trace((self.R + discount * self.B.T @ Ps[h + 1] @ self.B)
                                   * (policy_std ** 2)) for h in range(horizon))
        return - init_term - state_term - action_term

    # ------------------------------------------------- legacy scalar gradients
    def grad_K(self, K, Sigma):
        """Policy gradient wrt K (scalar A = B = I case only)."""
        I = np.eye(self.state_dim)
        if not np.array_equal(self.A, I) or not np.array_equal(self.B, I):
            raise NotImplementedError
        if not isinstance(K, Number) or not isinstance(Sigma, Number):
            raise NotImplementedError
        theta, sigma = float(K), float(Sigma)
        den = 1 - self.gamma * (1 + 2 * theta + theta ** 2)
        dePdeK = 2 * (theta * self.R / den
                      + self.gamma * (self.Q + theta ** 2 * self.R) * (1 + theta) / den ** 2)
        return float(- dePdeK * (self.max_pos ** 2 / 3 + self.gamma * sigma / (1 - self.gamma)))

    def grad_Sigma(self, K, Sigma=None):
        I = np.eye(self.state_dim)
        if not np.array_equal(self.A, I) or not np.array_equal(self.B, I):
            raise NotImplementedError
        if not isinstance(K, Number) or not isinstance(Sigma, Number):
            raise NotImplementedError
        P = self._computeP2(K)
        return float(-(self.R + self.gamma * P) / (1 - self.gamma))

    def grad_mixed(self, K, Sigma=None):
        I = np.eye(self.state_dim)
        if not np.array_equal(self.A, I) or not np.array_equal(self.B, I):
            raise NotImplementedError
        if not isinstance(K, Number) or not isinstance(Sigma, Number):
            raise NotImplementedError
        theta = float(K)
        den = 1 - self.gamma * (1 + 2 * theta + theta ** 2)
        dePdeK = 2 * (theta * self.R / den
                      + self.gamma * (self.Q + theta ** 2 * self.R) * (1 + theta) / den ** 2)
        return float(-dePdeK * self.gamma / (1 - self.gamma))

    def computeQFunction(self, x, u, K, Sigma, n_random_xn=100):
        """Monte-Carlo Q-value of (x, u) under a = K x + N(0, Sigma)."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        u = np.atleast_1d(np.asarray(u, dtype=float))
        if np.isscalar(Sigma):
            Sigma = float(Sigma) * np.eye(self.action_dim)
        Sigma = np.asarray(Sigma, dtype=float)

        P = self._computeP2(K)
        Qfun = 0.0
        for _ in range(n_random_xn):
            noise = self.sigma_noise @ self.np_random.standard_normal(self.state_dim)
            action_noise = self.np_random.multivariate_normal(
                np.zeros(Sigma.shape[0]), Sigma)
            nextstate = self.A @ x + self.B @ (u + action_noise) + noise
            Qfun -= (x @ self.Q @ x + u @ self.R @ u
                     + self.gamma * nextstate @ P @ nextstate
                     + (self.gamma / (1 - self.gamma))
                     * np.trace(Sigma @ (self.R + self.gamma * self.B.T @ P @ self.B)))
        return Qfun / n_random_xn

    # ------------------------------------------------------------------- render
    def render(self, mode='human', close=False):
        if self.state_dim not in [1, 2]:
            return
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        world_width = math.ceil((self.max_pos[0] * 2) * 1.5)
        xscale = screen_width / world_width
        ballradius = 3

        if self.state_dim == 1:
            screen_height = 400
        else:
            world_height = math.ceil((self.max_pos[1] * 2) * 1.5)
            screen_height = math.ceil(xscale * world_height)
            yscale = screen_height / world_height

        if self.viewer is None:
            clearance = 0
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            mass = rendering.make_circle(ballradius * 2)
            mass.set_color(.8, .3, .3)
            mass.add_attr(rendering.Transform(translation=(0, clearance)))
            self.masstrans = rendering.Transform()
            mass.add_attr(self.masstrans)
            self.viewer.add_geom(mass)
            if self.state_dim == 1:
                self.track = rendering.Line((0, 100), (screen_width, 100))
            else:
                self.track = rendering.Line((0, screen_height / 2), (screen_width, screen_height / 2))
            self.track.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(self.track)
            zero_line = rendering.Line((screen_width / 2, 0),
                                       (screen_width / 2, screen_height))
            zero_line.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(zero_line)

        x = self.state[0]
        ballx = x * xscale + screen_width / 2.0
        if self.state_dim == 1:
            bally = 100
        else:
            y = self.state[1]
            bally = y * yscale + screen_height / 2.0
        self.masstrans.set_translation(ballx, bally)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


if __name__ == '__main__':
    """Sanity check: optimal gain and return for the default 1-D system."""
    env = LQ(state_dim=1, action_dim=1, horizon=50, gamma=0.98)
    K_star = env.computeOptimalK()
    print('K^* =', K_star)
    print('J^* (infinite horizon) =', env.discounted_optimal_return(policy_std=0))
    print('J(K^*) via computeJ     =', env.computeJ(K_star, Sigma=0))

    n_episodes = 1000
    empirical_returns = []
    for i in range(n_episodes):
        state = env.reset()
        episode_return = 0
        gamma_t = 1.0
        
        for _ in range(env.horizon):
            action = np.dot(K_star, state)
            state, reward, done, _ = env.step(action)
            episode_return += gamma_t * reward
            gamma_t *= env.gamma
            if done:
                break
                
        empirical_returns.append(episode_return)
    print(f"Empirical Expected Return ({env.horizon} steps, N={n_episodes}): {np.mean(empirical_returns):.4f}")

