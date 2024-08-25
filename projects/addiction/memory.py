from collections import deque
import random
import numpy as np


class Memory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, health):
        self.buffer.append((state, action, reward, next_state, done, health))

    def sample(self, batch_size):
        state, action, reward, next_state, done, health = zip(*random.sample(self.buffer, k=batch_size))

        return np.array(state), action, reward, np.array(next_state), done, health

    def __len__(self):
        return len(self.buffer)
