from data_processors.base_processor import BaseProcessor
import numpy as np

# max state of pendulum
max_state = np.array([2.0, 0.07 ])
min_state = np.array([-2.0, -0.07 ])

class NormalizationDataProcessor(BaseProcessor):
    """Identity Data Processor, used for default values"""

    def __init__(self ) -> None:
        super().__init__()

    def transform(self, state):

        # Normalize the states between -1 and 1
        state = np.reshape(state, (2,))
        transformed_state = np.divide( ( (state - min_state) * 2 ), (max_state - min_state) ) - 1

        return transformed_state





