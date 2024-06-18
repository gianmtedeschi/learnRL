from hpo import HPO
from hyperopt import fmin, tpe, hp,  Trials
import numpy as np
import json
import os



optimizer= HPO(algorithm="VM",environment="pendulum",policy="split_gaussian",metric="average_performance",nni_flag=False)



lr=0.01
# Define the search space
space =  hp.loguniform('lr',-6,0)


# Run the optimization  # Change to False if you want to minimize the reward
trials = Trials()
best = fmin(
    fn= optimizer.objective,
    space=space,
    algo=tpe.suggest,
    max_evals=15,  # Number of evaluations
    trials=trials
)

results = []
for trial in trials.trials:
    result = {
        'learning_rate': trial['misc']['vals']['lr'][0],
        'loss': trial['result']['loss'],
        'status': trial['result']['status']
    }
    results.append(result)

json_file_path = os.path.join(optimizer.results_dir, f"trials_results.json")

with open(json_file_path, 'w') as jsonfile:
    json.dump(results, jsonfile, indent=4)

print("Best hyperparameters:", best)

