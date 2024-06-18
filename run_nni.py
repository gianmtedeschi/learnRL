import os
from nni.experiment import Experiment

# Set the working directory to the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Create an Experiment instance
experiment = Experiment('local')

# Configure the experiment
experiment.config.trial_command = 'python main_nni.py'
experiment.config.trial_code_directory = script_dir  # Ensure this is the correct directory
experiment.config.search_space_file = os.path.join(script_dir, 'search_space.json')
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args = {'optimize_mode': 'maximize'}
experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 2

# Start the experiment
experiment.run(8080)

# To view the experiment progress
print(f'NNI Experiment running at: http://localhost:8080')

# To stop the experiment after trials finish
# experiment.stop()
