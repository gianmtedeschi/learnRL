"""Implementation of ADAM optimizer"""

# imports
import numpy as np
import copy

# class
class Adam:
    def __init__(self, alpha=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m = np.array([0])
        self.v = np.array([0])
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.temp_updates = np.zeros((100, 2), dtype=object)
        self.ite = 0

    def next(self, grad, coord=None, local=False):
        if not local:
            self.t += 1

        m = copy.deepcopy(self.m)
        v = copy.deepcopy(self.v)

        if coord is None:
            m = self.beta1 * self.m + (1 - self.beta1) * grad
            v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        else: 
            m[coord] = self.beta1 * self.m[coord] + (1 - self.beta1) * grad
            v[coord] = self.beta2 * self.v[coord] + (1 - self.beta2) * grad ** 2

        m_hat = (m[coord] if coord is not None else m) / (1 - self.beta1 ** self.t)
        v_hat = (v[coord] if coord is not None else v) / (1 - self.beta2 ** self.t)

        # if grad.all() > 0:
        #     step = self.alpha / (np.sqrt(v_hat) + self.epsilon) * m_hat / grad   # division by grad needed since adam does not compute a step but an update vector!
        # else:
        #     step = 0
        step = self.alpha / (np.sqrt(v_hat) + self.epsilon) * m_hat

        if not local:
            self.m = copy.deepcopy(m)
            self.v = copy.deepcopy(v)
            self.ite = 0
        else:
            self.update_params(m, v, local=local)

        return step
    
    def update_params(self, m=None, v=None, local=False, coord=None, index=None):
        if local:
            self.temp_updates[self.ite] = [m[0],v[0]]
            self.ite += 1
        else:
            self.ite = 0
            upd_m = np.array((self.temp_updates[index*2][0], self.temp_updates[index*2 + 1][0]))
            upd_v = np.array((self.temp_updates[index*2][1], self.temp_updates[index*2 + 1][1]))

            self.m = np.concatenate([self.m[:coord], upd_m, self.m[coord + 1:]])
            self.v = np.concatenate([self.v[:coord], upd_v, self.v[coord + 1:]])

    def reset(self):
        self.m = 0
        self.v = 0
        self.t = 0

    def __str__(self):
        return 'Adam (alpha = %f, beta1 = %f, beta2 = %f, epsilon = %f)' % (
            self.alpha, self.beta1, self.beta2, self.epsilon)