from data_processors.base_processor import BaseProcessor
import numpy as np
import itertools

p1_NUM_RBF = 15
p2_NUM_RBF = 15
v_NUM_RBF = 15

NUM_STATES = p1_NUM_RBF * p2_NUM_RBF * v_NUM_RBF
NUM_ACTIONS = 1
INIT_SIGMA = 0.5
TERM_SIGMA = 0.1

p1_mu = np.linspace(-1, 1, p1_NUM_RBF)
p2_mu = np.linspace(-1, 1, p2_NUM_RBF)
v_mu = np.linspace(-1, 1, v_NUM_RBF)
MUS = np.array(list(itertools.product(p1_mu, p2_mu, v_mu)))

# max state of pendulum
max_state = np.array([1.0, 1.0, 8.0 ])
min_state = np.array([-1.0, -1.0, -8.0 ])
rbf_sigma = 0.1

class KernelDataProcessor(BaseProcessor):
    """Identity Data Processor, used for default values"""

    def __init__(self ) -> None:
        super().__init__()
        self.num_states = NUM_STATES

    def transform(self, state):
        output = np.zeros((NUM_STATES, NUM_ACTIONS))

        # Normalize the states between -1 and 1
        state = np.reshape(state, (3,))
        state = np.divide( ( (state - min_state) * 2 ), (max_state - min_state) ) - 1

        # compute the mean of the states
        state_mu = np.linalg.norm(np.reshape(state, (len(state),)) - MUS, 2, axis=1)
        
        # compute how far away the original states are from the radial basis function kernel centers
        output = np.array(np.exp(- ( state_mu ) **2 / ( 2*(rbf_sigma**2) )) / (rbf_sigma * np.sqrt(2 * np.pi)))
        return output




# def rbf_transform(state):
#     """ Applies the radial basis function transformation on the raw states to convert to
#         discrete states
    

#     Args:
#         state (list): raw state from the openai gym environment

#     Returns:
#         output (numpy.array): the transformed state in the (NUM_STATES, NUM_ACTIONS) dimensions
#     """
#     output = np.zeros((NUM_STATES, NUM_ACTIONS))

#     # Normalize the states between -1 and 1
#     state = np.reshape(state, (3,))
#     state = np.divide( ( (state - min_state) * 2 ), (max_state - min_state) ) - 1

#     # compute the mean of the states
#     state_mu = np.linalg.norm(np.reshape(state, (len(state),)) - MUS, 2, axis=1)
    
#     # compute how far away the original states are from the radial basis function kernel centers
#     output = np.array(np.exp(- ( state_mu ) **2 / ( 2*(rbf_sigma**2) )) / (rbf_sigma * np.sqrt(2 * np.pi)))
#     return output
