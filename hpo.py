

# Libraries
import copy

from envs import *
from policies import GaussianPolicy, SplitGaussianPolicy, LinearPolicy, DeepGaussian
from algorithms import PolicyGradient
from algorithms import PolicyGradientSplitMultiDim
from algorithms import PolicyGradientSplitMultiDimVM
from data_processors import IdentityDataProcessor
from common.tree import BinaryTree, Node
from art import *
import datetime
import torch.nn as nn
import json
import os
import glob
from hyperopt import STATUS_OK

class HPO :
    def __init__(self,algorithm:str="VM",
                 environment:str="lq_1",
                 policy:str="split_gaussian",
                 metric: str="average_performance",
                 nni_flag: bool=False
                 ) -> None:
        
        err_msg="Algorithm must be valid"
        assert algorithm in ["pg","VM","Bernstein"],err_msg
        self.algorithm=algorithm

        err_msg="Environment must be valid"
        assert environment in ["lq_1","lq_2","minigolf","half_cheetah","swimmer","pendulum"],err_msg
        self.environment=environment

        err_msg="Policy must be valid "
        assert policy in ["split_gaussian","gaussian","linear","nn"],err_msg
    

        err_msg="Policy must be valid matchup"
        assert (algorithm=="pg" and policy in ["gaussian","linear","nn"]) or (algorithm!="pg" and policy=="split_gaussian"),err_msg
        self.policy=policy

        err_msg="Metric must be valid"
        assert metric in ["average_performance","best_performance"],err_msg
        self.metric=metric

        self.nni_flag=nni_flag

        
        self.env= None
        self.pol= None
        self.alg=None
        self.tot_params=0

        # Directory name based on current time
        self.dir = "hpo_test" + "_" + self.environment + "_" + datetime.datetime.now().strftime("%y_%d_%m-%H_%M_")

        # Create a directory for results
        if self.algorithm == "pg":
            base_dir = "/Users/Admin/OneDrive/Documenti/GitHub/learnRL/hpo/pg/"
        elif self.algorithm == "Bernstein":
            base_dir = "/Users/Admin/OneDrive/Documenti/GitHub/learnRL/hpo/Bernstein/"
        elif self.algorithm == "VM":
            base_dir = "/Users/Admin/OneDrive/Documenti/GitHub/learnRL/hpo/VM/"

        self.results_dir = os.path.join(base_dir, self.dir)
        os.makedirs(self.results_dir, exist_ok=True)


    
    def env_initialization(self):
        if self.environment == "lq_1":
             
             self.env = LQ(horizon=10, gamma=0.999,action_dim=1, state_dim=1)
        
        elif self.environment=="lq_2":
             
             self.env= LQ(horizon=10,gamma=0.999,action_dim=2,state_dim=2)

        elif self.environment== "pendulum":
             
             self.env= PendulumEnv(horizon=100)
        
        elif self.environment=="swimmer":
             
             self.env= Swimmer(horizon=100,gamma=0.995)
        
        elif self.environment=="half_cheetah":
             
             self.env= HalfCheetah(horizon=100,gamma=0.995)
        
        elif self.environment=="minigolf":

             self.env= MiniGolf(horizon=50,gamma=0.995)

    def pol_initialization(self):

        self.tot_params= self.env.state_dim*self.env.action_dim
        state_dim=self.env.state_dim
        action_dim=self.env.action_dim
         
        if self.policy=="linear":
              self.pol= LinearPolicy(parameters=np.ones(self.tot_params),
                                     dim_state=state_dim,
                                     dim_action=action_dim,
                                     multi_linear=True)
        
        elif self.policy=="gaussian":
             self.pol= GaussianPolicy(parameters=np.ones(self.tot_params),
                                      dim_state=state_dim,
                                      dim_action=action_dim,
                                      std_dev=0.1,
                                      std_decay=0,
                                      std_min=1e-6,
                                      multi_linear=True)
        
        elif self.policy=="nn":
             net = nn.Sequential(nn.Linear(state_dim, 16, bias=False),
                                 nn.Tanh(),nn.Linear(16,8,bias=False),
                                 nn.Tanh(),nn.Linear(8,action_dim,bias=False))
             model_desc= dict(layers_shape=[(state_dim,16),(16,8),(8,action_dim)])

             self.pol=DeepGaussian(parameters=None,
                                   input_size=state_dim,
                                   output_size=action_dim,
                                   model=copy.deepcopy(net),
                                   model_desc=copy.deepcopy(model_desc),
                                   std_dev=0.1,
                                   std_decay=0,
                                   std_min=1e-6)
             
             
        
        elif self.policy=="split_gaussian":
             self.tot_params=action_dim
             self.pol = SplitGaussianPolicy(parameters=np.ones(self.tot_params),
                                            dim_state=state_dim,
                                            dim_action=action_dim,
                                            std_dev=0.1,
                                            std_decay=0,
                                            std_min=1e-6,history=BinaryTree())
             

    def alg_initialization(self,learning_rate):
         
        if self.algorithm=="pg":
              
            subdir_path = os.path.join(self.results_dir,"hpo_test" +"_"+ self.policy + "_" + datetime.datetime.now().strftime("%y_%d_%m-%H_%M_"))
            os.makedirs(subdir_path, exist_ok=True)
            alg_parameters=dict(lr=[learning_rate],
                                  lr_strategy="adam",
                                  estimator_type="GPOMDP",
                                  initial_theta=[0]*self.tot_params,
                                  ite=1000,
                                  batch_size=100,
                                  env=self.env,
                                  policy=self.pol,
                                  data_processor=IdentityDataProcessor(),
                                  directory=subdir_path,
                                  verbose=False,
                                  natural=False,
                                  checkpoint_freq=100,
                                  n_jobs=8,
                                  baselines=None)
            self.alg=PolicyGradient(**alg_parameters)

        if self.algorithm=="Bernstein":
               
            subdir_path = os.path.join(self.results_dir,"hpo_test"+ "_" + datetime.datetime.now().strftime("%y_%d_%m-%H_%M_"))
            os.makedirs(subdir_path, exist_ok=True)
            alg_parameters=dict(lr=[learning_rate],
                                  lr_strategy="adam",
                                  estimator_type="GPOMDP",
                                  initial_theta=[0]*self.tot_params,
                                  ite=1000,
                                  batch_size=100,
                                  env=self.env,
                                  policy=self.pol,
                                  data_processor=IdentityDataProcessor(),
                                  directory=subdir_path,
                                  verbose=False,
                                  natural=False,
                                  checkpoint_freq=100,
                                  n_jobs=8,
                                  baselines="peters",
                                  split_grid=None)
            self.alg = PolicyGradientSplitMultiDim(**alg_parameters)

        if self.algorithm=="VM":

            subdir_path = os.path.join(self.results_dir,"hpo_test"+ "_" + datetime.datetime.now().strftime("%y_%d_%m-%H_%M_"))
            os.makedirs(subdir_path, exist_ok=True)
            alg_parameters=dict(lr=[learning_rate],
                                  lr_strategy="adam",
                                  estimator_type="GPOMDP",
                                  initial_theta=[0] * self.tot_params,
                                  ite=1000,
                                  batch_size=100,
                                  env=self.env,
                                  policy=self.pol,
                                  data_processor=IdentityDataProcessor(),
                                  directory=subdir_path,
                                  verbose=False,
                                  natural=False,
                                  checkpoint_freq=50,
                                  n_jobs=8,
                                  baselines="peters",
                                  split_grid=None)
            self.alg=PolicyGradientSplitMultiDimVM(**alg_parameters)


    def extract_metric(self,results_file_path):

        try:
         
            # Find all JSON files in the specified folder
            json_files = glob.glob(os.path.join(results_file_path, '*.json'))
        
            # Check if exactly one JSON file is found
            if len(json_files) != 1:
               raise FileNotFoundError("The folder must contain exactly one JSON file.")
        
            # Open and read the JSON file
            json_file_path = json_files[0]
            with open(json_file_path, 'r') as file:
               data = json.load(file)
            
        
            if self.metric=="best_performance":
                if "best_perf" in data:
                     metric= data["best_perf"]
                else:
                     raise KeyError("Key 'best_perf' not found in JSON data")
                
            elif self.metric=="average_performance":

                if "performance" in data:
                    performance=data["performance"]
                    if isinstance(performance, list) and all(isinstance(i, (int, float)) for i in performance):
                         metric = sum(performance) / len(performance) if performance else 0
                    else:
                         raise ValueError("The 'performance' key must contain a list of numerical values.")
                else:
                     raise KeyError("Key 'performance' not found in JSON data")
                
                         

                
                 
            # Return the extracted information
            return metric

        except FileNotFoundError:
            print(f"Error: The file {results_file_path} was not found.")
            return None
    
        except json.JSONDecodeError:
            print(f"Error: The file {results_file_path} is not a valid JSON file.")
            return None
    
        except KeyError as e:
            print(f"Error: Missing key in JSON data - {e}")
            return None
    
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
         
        


        
    
       



    
    def objective(self,lr):
         
    
       # Initialize the environment and the RL agent
       self.env_initialization()
       self.pol_initialization()
       self.alg_initialization(learning_rate=lr)  # Replace with your agent's initialization
       self.alg.learn()
       path= self.alg.directory
       metric= self.extract_metric(path)
       # Save intermediate results
       with open(os.path.join(self.results_dir, 'intermediate_results.json'), 'w') as f:
            json.dump({"learning_rate": lr, "metric": metric}, f, indent=4)
       # Adjust return value based on whether we want to maximize or minimize

       if  not self.nni_flag:
            return {'loss': -metric,'status':STATUS_OK}
       else:
           return metric
            
       
       


        

