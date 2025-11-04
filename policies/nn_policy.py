from abc import ABC
from policies import BasePolicy
import torch
import torch.nn as nn
import numpy as np
import math

class NeuralNetworkPolicy(BasePolicy, ABC):
    def __init__(self,
                 parameters: np.array = None,
                 input_size: int = 1, 
                 output_size: int = 1,
                 model: nn.Sequential = None):
        
        super().__init__()

        self.parameters = parameters

        self.dim_state = input_size
        self.dim_action = output_size
        
        self.net = None
        self.layers_shape = None

        # Build default model
        if model is None:
            self.net = nn.Sequential(
                nn.Linear(self.dim_state, 16, bias=True),
                nn.Linear(16, 16, bias=True),
                nn.Linear(16, self.dim_action, bias=False)
            )
            
        else:
            self.net = model
        
        self._refresh_param_index()

        if self.parameters is None:
            self.parameters = self.get_parameters()
        
        self.set_parameters(self.parameters)

        
    def _refresh_param_index(self):
        self.param_shapes = [ tuple(p.shape) for p in self.net.parameters()]
        self.param_sizes = [ int(np.prod(t)) for t in self.param_shapes]
        self.param_idx = np.cumsum(self.param_sizes)
        self.tot_params = int(sum(self.param_sizes))
    
    def draw_action(self, state):
        tensor_state = torch.tensor(np.array(state, dtype=np.float64)).unsqueeze(0)
        action = np.array(torch.detach(self.net(tensor_state)))
        return action

    def reduce_exploration(self):
        raise NotImplementedError("[NNPolicy] Ops, not implemented yet!")
    
    def get_parameters(self):
        theta = []
        for i, param_layer in enumerate(self.net.parameters()):
            theta += list(param_layer.data.numpy().flatten())
        return np.array(theta)
    

    def set_parameters(self, thetas) -> None:
        # check on the number of parameters
        err_msg = f"[NNPolicy] Number of parameters {len(thetas)} is different from "
        err_msg += f"{self.tot_params}"
        assert len(thetas) == np.sum(self.tot_params), err_msg

        # set the weights
        tensor_param = torch.tensor(np.array(thetas, dtype=np.float64))
        for i, param_layer in enumerate(self.net.parameters()):
            if i == 0:
                batch_params = tensor_param[: self.param_idx[i]]
            elif i == len(self.param_shapes) - 1:
                batch_params = tensor_param[self.param_idx[i - 1]:]
            else:
                batch_params = tensor_param[self.param_idx[i - 1]:self.param_idx[i]]
            reshaped_params = torch.reshape(batch_params, self.param_shapes[i])
            param_layer.data = nn.parameter.Parameter(reshaped_params, requires_grad=True)



    def compute_score(self, state, action) -> np.array:
        return np.zeros(self.tot_params)
    
    def compute_logprob(self, state, action):
        raise NotImplementedError("compute_logprob not impelmented for this policy")

    


class DeepGaussianPolicy(NeuralNetworkPolicy):
    def __init__(
            self, parameters: np.array = None,
            input_size: int = 1,
            output_size: int = 1,
            model: nn.Sequential = None,
            std_dev: float = 1,
            std_decay: float = 0,
            std_min: float = 1e-6
    ) -> None:
        super().__init__(
            parameters=parameters,
            input_size=input_size,
            output_size=output_size,
            model=model,
        )
        self.std_dev = std_dev
        self.std_decay = std_decay
        self.std_min = std_min

    def compute_score(self, state, action) -> np.array:
        # if state.ndim == 2:
        #     state = state.ravel()
        # Convert state and action to tensors
        state_tensor = torch.tensor(np.array(state, dtype=np.float64)).unsqueeze(0)
        action_tensor = torch.tensor(np.array(action, dtype=np.float64)).unsqueeze(0)

        # Standard deviation (as tensor)
        sigma_tensor = torch.tensor(self.std_dev, dtype=torch.float64)

        # Forward pass to compute the mean action
        action_mean = self.net.forward(state_tensor)
        
        #log_prob = -0.5 * (((action_tensor - action_mean) / sigma_tensor) ** 2).sum() - 0.5 * torch.log(torch.sqrt(2 * torch.pi * sigma_tensor ** 2)) * action_tensor.size(0)
        log_prob = -0.5 * (((action_tensor - action_mean) / sigma_tensor) ** 2).sum() - 0.5 * torch.log(2 * torch.pi * sigma_tensor ** 2) * action_tensor.size(0)

        # log_prob = -((action_tensor - action_mean) ** 2) / (2 * sigma_tensor ** 2) - sigma_tensor - .5 * math.log(2 * math.pi)
        # log_prob = -0.5 * ((action_tensor - action_mean) / sigma_tensor).pow(2).sum()
        # log_prob -= 0.5 * torch.log(2 * torch.pi * sigma_tensor ** 2) * action_tensor.size(0)

        # print(state_tensor, action_tensor, action_mean, log_prob)
        # Zero out gradients from the previous pass
        self.net.zero_grad()

        # Compute gradients of the log-probability w.r.t. network parameters
        log_prob.backward()

        # for param_layer in self.net.parameters():
        #     print(param_layer.grad)

        # Collect gradients layer by layer and flatten into a single vector
        grads = np.zeros(self.tot_params, dtype=np.float64)
        for i, param_layer in enumerate(self.net.parameters()):
            layer_grads = param_layer.grad.detach().numpy().ravel()  # Detach and flatten
            if i == 0:
                grads[:self.param_idx[i]] = layer_grads
            elif i == len(self.param_shapes) - 1:
                grads[self.param_idx[i - 1]:] = layer_grads
            else:
                grads[self.param_idx[i - 1]:self.param_idx[i]] = layer_grads

        return grads

    def reduce_exploration(self):
        self.std_dev = np.clip(
            self.std_dev - self.std_decay,
            self.std_min,
            np.inf,
            dtype=np.float64
        )

    def draw_action(self, state) -> np.array:
        means = np.array(super().draw_action(state=state), dtype=np.float64)
        # action = np.array(np.random.normal(means, self.std_dev), dtype=np.float64)
        action = np.array(
            means + self.std_dev * np.random.normal(0, 1, self.dim_action),
            dtype=np.float64
        )
        return action.ravel()
    
    def compute_logprob(self, state, action):
        #if state.ndim == 2:
            #state = state.ravel()
        state_tensor = torch.tensor(np.array(state, dtype=np.float64)).unsqueeze(0)
        action_tensor = torch.tensor(np.array(action, dtype=np.float64)).unsqueeze(0)
        
        sigma_tensor = torch.tensor(self.std_dev, dtype=torch.float64)
        
        action_mean = self.net.forward(state_tensor)
        
        #log_prob = -0.5 * (((action_tensor - action_mean) / sigma_tensor) ** 2).sum() - 0.5 * torch.log(torch.sqrt(2 * torch.pi * sigma_tensor ** 2)) * action_tensor.size(0)
        log_prob = -0.5 * (((action_tensor - action_mean) / sigma_tensor) ** 2).sum() - 0.5 * torch.log(2 * torch.pi * sigma_tensor ** 2) * action_tensor.size(0)


        return log_prob
    
    
    def compute_logprob_batch(self, states, actions):
        
        """
        Computes log-probabilities for batched trajectories under a diagonal Gaussian policy.

        Args:
            states: Tensor of shape (batch_size, horizon, state_dim)
            actions: Tensor of shape (batch_size, horizon, action_dim)

        Returns:
            log_probs: Tensor of shape (batch_size, horizon), one per timestep per sample
        """
        if not torch.is_tensor(states):
            states = torch.as_tensor(states, dtype=torch.float64)
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.float64)

        B, H, D = states.shape
        A = actions.shape[-1]

        # Flatten batch and time dimensions to feed into the network
        states_flat = states.view(B * H, D)

        # Get action mean from policy network
        action_mean = self.net.forward(states_flat)  # shape: (B*H, action_dim)

        # Get std (broadcastable shape)
        std = torch.as_tensor(self.std_dev, dtype=torch.float64)
        if std.ndim == 0:
            std = std.unsqueeze(0)  # make it 1D
        var = std ** 2

        # Flatten actions for matching
        actions_flat = actions.view(B * H, A)

        # Compute log probs
        log_probs = -0.5 * (((actions_flat - action_mean) ** 2) / var + 2 * torch.log(std) + torch.log(torch.tensor(2 * torch.pi)))
        log_probs = log_probs.sum(dim=1)  # shape: (B*H,)

        # Reshape back to (B, H)
        log_probs = log_probs.view(B, H)

        return log_probs
            