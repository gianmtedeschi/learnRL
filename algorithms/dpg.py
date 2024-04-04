import numpy as np

class DeterministicPolicyGradient:
    def __init__(self, policy, learning_rate=0.01, discount_factor=0.99):
        self.policy = policy
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def update(self, states, actions, rewards, next_states):
        # Compute the action gradients
        action_gradients = self.compute_action_gradients(states, actions)

        # Update the policy parameters
        self.policy.update_parameters(states, action_gradients, self.learning_rate)

        # Update the value function using the TD error
        td_errors = rewards + self.discount_factor * self.policy.compute_value(next_states) - self.policy.compute_value(states)
        self.policy.update_value_function(states, td_errors)

    def compute_action_gradients(self, states, actions):
        action_gradients = []
        for state, action in zip(states, actions):
            # Compute the gradient of the action with respect to the policy parameters
            gradient = self.policy.compute_action_gradient(state, action)
            action_gradients.append(gradient)
        return action_gradients