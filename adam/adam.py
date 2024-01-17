"""Implementation of ADAM optimizer"""

# imports
import numpy as np

# class
class Adam:
    def __init__(self, alpha=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m = 0
        self.v = 0
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def next(self, grad, *args):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        # if grad.all() > 0:
        #     step = self.alpha / (np.sqrt(v_hat) + self.epsilon) * m_hat / grad   # division by grad needed since adam does not compute a step but an update vector!
        # else:
        #     step = 0
        step = self.alpha / (np.sqrt(v_hat) + self.epsilon) * m_hat
        return step

    def reset(self):
        self.m = 0
        self.v = 0
        self.t = 0

    def __str__(self):
        return 'Adam (alpha = %f, beta1 = %f, beta2 = %f, epsilon = %f)' % (
            self.alpha, self.beta1, self.beta2, self.epsilon)