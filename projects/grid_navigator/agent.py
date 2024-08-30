from collections import deque
from dcqn import DCQN
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


class Agent:
    def __init__(self, state_shape, action_size, gamma=0.99, lr=0.001, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, memory_size=10000, tau=0.001):
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.epsilon_min = epsilon_min  # Minimum epsilon value
        self.memory = deque(maxlen=memory_size)  # Experience replay memory
        self.tau = tau  # Parameter for soft update of target network

        # Initialize the main model and the target model
        self.model = DCQN(state_shape, action_size)
        self.target_model = DCQN(state_shape, action_size)
        self.update_target_model()  # Ensure target model starts with the same weights

        # Optimizer for training the network
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def update_target_model(self):
            # Hard update: Copy weights from the main model to the target model
            self.target_model.load_state_dict(self.model.state_dict())

    def soft_update_target_model(self):
        # Soft update: Slowly blend target model weights towards the main model
        for target_param, online_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

    def act(self, state):
        # Choose an action using epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore: choose a random action
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()  # Exploit: choose the best action based on current policy

    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay memory
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        # Train the model using a batch of experiences from replay memory
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                # Compute the target Q-value using the target network
                next_action = torch.argmax(self.model(torch.FloatTensor(next_state).unsqueeze(0))).item()
                target += self.gamma * self.target_model(torch.FloatTensor(next_state).unsqueeze(0))[0][next_action].item()

            # Get the current Q-value estimates and update them
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0)).clone().detach()
            target_f[0][action] = target

            # Perform a gradient descent step to update the main network
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(torch.FloatTensor(state).unsqueeze(0)), target_f)
            loss.backward()
            self.optimizer.step()

        # Decay epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
