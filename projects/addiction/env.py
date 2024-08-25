import numpy as np
import math
import sys
import pygame


class AddictionEnv:
    def __init__(self, init_state: bool = True):
        self.size = 15
        self.agent_pos = None
        self.state = None
        self.done = None
        self.health = 1.0
        self.nearby_reward_pos = (12, 4)
        self.far_reward_pos = (4, 10)
        self.screen = None
        self.cell_size = 60
        self.font = None

        if init_state: self.reset()

    def reset(self):
        self.state = np.zeros((self.size, self.size), dtype=int)
        self.agent_pos = (self.size - 1, self.size // 2)
        self.done = False

        self.state[self.agent_pos] = 1

        self.state[self.nearby_reward_pos] = 2
        self.state[self.far_reward_pos] = 3

        return self.state

    def step(self, action):
        if self.done:
            return self.state, 0, self.done, {}

        moves = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

        new_pos = (
            self.agent_pos[0] + moves[action][0],
            self.agent_pos[1] + moves[action][1]
        )

        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            self.agent_pos = new_pos

        reward = 0
        if self.agent_pos == self.nearby_reward_pos:
            reward = 250
            self.health = self.health * 0.935
            self.done = True
        elif self.agent_pos == self.far_reward_pos:
            reward = 140
            self.health = self.health * 1.5
            self.done = True
        else:
            reward = -0.1

        self.health = math.floor(self.health * 10) / 10

        if self.health <= 0:
            self.done = True
            reward = -2500
            self.health = 1.0

        self.state = np.zeros((self.size, self.size), dtype=int)
        self.state[self.agent_pos] = 1
        self.state[self.nearby_reward_pos] = 2
        self.state[self.far_reward_pos] = 3

        return self.state, reward, self.done, {'health': self.health}

    def render(self, episode, num_episodes, reward, health, eps):
        if self.screen is None:
            pygame.init()
            screen_size = self.size * self.cell_size
            self.screen = pygame.display.set_mode((screen_size, screen_size))
            pygame.display.set_caption("AddictionEnv Game")
            self.font = pygame.font.Font(None, 36)

        colors = {
            0: (255, 255, 255),
            1: (0, 0, 255),
            2: (255, 0, 0),
            3: (0, 255, 0)
        }

        for row in range(self.size):
            for col in range(self.size):
                color = colors[self.state[row, col]]
                pygame.draw.rect(self.screen, color, pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size), 1)

        episodes_text = self.font.render(f'Episodes: [{episode}/{num_episodes}]', True, (20, 20, 20))
        current_reward_text = self.font.render(f'Reward: {reward:.2f}', True, (20, 20, 20))
        health_text = self.font.render(f'Health: {health:.2f}', True, (20, 20, 20))
        eps_text = self.font.render(f'Epsilon: {eps:.5f}', True, (20, 20, 20))

        self.screen.blit(episodes_text, (0, 0))
        self.screen.blit(current_reward_text, (0, 20))
        self.screen.blit(health_text, (0, 40))
        self.screen.blit(eps_text, (0, 60))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
