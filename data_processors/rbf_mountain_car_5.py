from data_processors.base_processor import BaseProcessor
import numpy as np
import itertools

s_NUM_RBF = 40
v_NUM_RBF = 15

NUM_STATES = s_NUM_RBF* v_NUM_RBF
NUM_ACTIONS = 1

s_mu = np.linspace(-1, 1, s_NUM_RBF)
v_mu = np.linspace(-1, 1, v_NUM_RBF)
MUS = np.array(list(itertools.product(s_mu, v_mu)))

# max state of mountain car simm 
max_state = np.array([4.5, 0.07 ])
min_state = np.array([-4.5, -0.07 ])
rbf_sigma = 0.1

class RBFMountainCar_v5DataProcessor(BaseProcessor):
    """Identity Data Processor, used for default values"""

    def __init__(self ) -> None:
        super().__init__()
        self.num_states = NUM_STATES

    def transform(self, state):
        output = np.zeros((NUM_STATES, NUM_ACTIONS))

        # Normalize the states between -1 and 1
        state = np.reshape(state, (2,))
        state = np.divide( ( (state - min_state) * 2 ), (max_state - min_state) ) - 1

        # compute the mean of the states
        state_mu = np.linalg.norm(np.reshape(state, (len(state),)) - MUS, 2, axis=1)
        
        # compute how far away the original states are from the radial basis function kernel centers
        output = np.array(np.exp(- ( state_mu ) **2 / ( 2*(rbf_sigma**2) )) / (rbf_sigma * np.sqrt(2 * np.pi)))
        return output





