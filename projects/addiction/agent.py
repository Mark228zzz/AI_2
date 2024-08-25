from memory import Memory
from network import Network
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class Agent:
    def __init__(self, state_size, action_size, buffer_size=10000, batch_size=64, gamma=0.99, lr=0.001, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = Memory(buffer_size) # Replay memory to store experiences
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.epsilon_min = epsilon_min  # Minimum value for epsilon
        self.batch_size = batch_size  # Batch size for training
        self.learning_rate = lr  # Learning rate for the optimizer
        self.tau = 0.001 # Tau value for soft update

        # Initialize the primary and target networks
        self.model = Network(state_size, action_size)
        self.target_model = Network(state_size, action_size)
        self.soft_update_target_model(self.target_model, self.model, self.tau)

        # Set up the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def soft_update_target_model(self, target_model, online_model, tau=0.001):
        # Soft update the params of target model
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def act(self, state):
        # Choose an action using epsilon-greedy policy
        if np.random.rand() <= self.epsilon: # EXPLORATION
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state) # EXPLOITATION
        return torch.argmax(act_values, dim=1).item()

    def remember(self, state, action, reward, next_state, done, health):
        self.memory.add(state, action, reward, next_state, done, health)

    def train(self):
        # Train the model by sampling a batch from replay memory
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch from memory
        states, actions, rewards, next_states, dones, health = self.memory.sample(self.batch_size)

        # Convert to torch tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute current Q values
        current_q = self.model(states).gather(1, actions)

        # Compute target Q values
        next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
        target_q = (rewards + (self.gamma * next_q * (1 - dones)))*torch.tensor(health)

        # Compute loss and update the network
        loss = nn.MSELoss()(current_q, target_q) # Calculate the loss
        self.optimizer.zero_grad() # Zero gradient
        loss.backward() # Backpropogate the loss
        self.optimizer.step() # Update parameters

        # Decay epsilon after each episode
        self.update_eps()

    def update_eps(self):
        # Update epsilon with decay, ensuring it does not fall below epsilon_min
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
