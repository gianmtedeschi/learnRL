import numpy as np

class ContinuousActionMDP:
    def __init__(self, H=100, W=5, epsilon=0.2):
        self.H = H
        self.W = W
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.state = 0.0
        self.step_count = 0
        self.done = False
        return self.state

    def step(self, action):
        if self.done:
            return self.state, 0.0, True, {}

        if self.step_count % self.W == 0 and self.step_count > 0:
            # stochastic transition
            if np.random.rand() < self.epsilon:
                self.done = True
                return -1.0, 0.0, True, {}  # deviant state, negative state value
            else:
                self.state += action
        else:
            self.state += action

        reward = action
        self.step_count += 1

        if self.step_count >= self.H:
            self.done = True

        return self.state, reward, self.done, {}

    def render(self):
        print(f"Step {self.step_count}, State: {self.state}")


class ContinuousObservationNoiseMDP:
    """
    Continuous action MDP where observations are noisy except every W steps.
    """

    def __init__(self, H=100, W=5, epsilon=0.2, noise_std=0.1):
        self.H = H                # Horizon
        self.W = W                # Clean observation frequency
        self.epsilon = epsilon    # Probability of stochastic transition
        self.noise_std = noise_std  # Std dev of observation noise
        self.reset()

    def reset(self):
        self.state = 0.0
        self.step_count = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        if self.step_count % self.W == 0:
            return self.state  # Clean observation
        else:
            return self.state + np.random.normal(0, self.noise_std)

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, True, {}

        # Stochastic jump at periodic intervals
        if self.step_count % self.W == 0 and self.step_count > 0:
            if np.random.rand() < self.epsilon:
                self.done = True
                return -1.0, 0.0, True, {}
            else:
                self.state += action
        else:
            self.state += action

        reward = action
        self.step_count += 1
        if self.step_count >= self.H:
            self.done = True

        return self._get_observation(), reward, self.done, {}

    def render(self):
        print(f"Step {self.step_count}, True state: {self.state}, Observation: {self._get_observation()}")



# env = ContinuousObservationNoiseMDP()

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