from hpo import HPO
from hyperopt import fmin, tpe, hp,  Trials
import numpy as np
import json
import os
import nni



optimizer= HPO(algorithm="pg",environment="lq_2",policy="linear",metric="average_performance")



# Fetch hyperparameters from NNI
params = nni.get_next_parameter()
learning_rate = params['learning_rate']

# Create an instance of HPO
optimizer= HPO(algorithm="pg",environment="lq_2",policy="linear",metric="average_performance",nni_flag=True)

# Run the objective function with the provided learning rate
result = optimizer.objective(learning_rate, maximize=False)

# Report the result back to NNI
nni.report_final_result(result)


