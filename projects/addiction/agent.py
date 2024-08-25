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
        self.memory = Memory(buffer_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.learning_rate = lr

        self.model = Network(state_size, action_size)
        self.target_model = Network(state_size, action_size)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values, dim=1).item()

    def remember(self, state, action, reward, next_state, done, health):
        self.memory.add(state, action, reward, next_state, done, health)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, health = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q = self.model(states).gather(1, actions)

        next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
        target_q = (rewards + (self.gamma * next_q * (1 - dones)))*torch.tensor(health)

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_eps()

    def update_eps(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_values(self, state):
        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.squeeze().cpu().numpy()
