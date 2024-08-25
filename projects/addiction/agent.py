from env import AddictionEnv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import random
import math


class Network(nn.Module):
    def __init__(self, state_size, action_size):
        super(Network, self).__init__()

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Memory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, health):
        self.buffer.append((state, action, reward, next_state, done, health))

    def sample(self, batch_size):
        state, action, reward, next_state, done, health = zip(*random.sample(self.buffer, k=batch_size))

        return np.array(state), action, reward, np.array(next_state), done, health

    def size(self):
        return len(self.buffer)


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
        if self.memory.size() < self.batch_size:
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

env = AddictionEnv()
state_size = env.size ** 2
action_size = 4

agent = Agent(state_size, action_size)

episodes = 1000
update_target_every = 10
max_steps = 200

for e in range(episodes):
    state = env.reset().flatten()
    total_reward = 0

    for time in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        health = info['health']
        next_state = next_state.flatten()
        agent.remember(state, action, reward, next_state, done, health)
        state = next_state
        total_reward += reward

        if done:
            break

        agent.replay()

        env.render(e, episodes, total_reward, health, agent.epsilon)

    agent.update_target_model()
    print(f"Episode {e+1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}, Health: {health}")

    if e % update_target_every == 0:
        agent.update_target_model()

env.close()
