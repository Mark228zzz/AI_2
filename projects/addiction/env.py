import numpy as np
import math
import sys
import pygame


class AddictionEnv:
    def __init__(self, init_state: bool = True):
        # Initialize the environment with a default size of 15x15
        self.size = 15
        self.agent_pos = None  # Position of the agent in the environment
        self.state = None  # 2D grid representing the environment
        self.done = None  # Flag to indicate if the episode has ended
        self.health = 1.0  # Initial health of the agent
        self.nearby_reward_pos = (12, 4)  # Position of the nearby reward
        self.far_reward_pos = (4, 10)  # Position of the far reward
        self.screen = None  # Pygame screen for rendering
        self.cell_size = 60  # Size of each cell in the grid
        self.font = None  # Font used for rendering text in pygame

        if init_state: self.reset()  # Reset the environment to its initial state

    def reset(self):
        # Reset the environment's state, agent's position, and health
        self.state = np.zeros((self.size, self.size), dtype=int)
        self.agent_pos = (self.size - 1, self.size // 2)  # Start the agent at the bottom-center
        self.done = False  # Set done flag to False, indicating the episode is active

        # Place the agent, nearby reward, and far reward in the grid
        self.state[self.agent_pos] = 1
        self.state[self.nearby_reward_pos] = 2
        self.state[self.far_reward_pos] = 3

        return self.state

    def step(self, action):
        if self.done:
            return self.state, 0, self.done, {}

        # Define possible moves: 0 = up, 1 = down, 2 = left, 3 = right
        moves = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

        # Calculate the new position of the agent based on the action taken
        new_pos = (
            self.agent_pos[0] + moves[action][0],
            self.agent_pos[1] + moves[action][1]
        )

        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            self.agent_pos = new_pos

        reward = 0
        if self.agent_pos == self.nearby_reward_pos:
            # If the agent reaches the nearby reward position
            reward = 250
            self.health *= 0.935  # Decrease health
            self.done = True  # End the episode
        elif self.agent_pos == self.far_reward_pos:
            # If the agent reaches the far reward position
            reward = 50
            self.health *= 1.5  # Increase health
            self.done = True  # End the episode
        else:
            reward = 0.0  # No reward for an empty cell

        # Ensure health is rounded down to one decimal place
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
        if self.screen is None: # If there`s no screen
            pygame.init()
            screen_size = self.size * self.cell_size
            self.screen = pygame.display.set_mode((screen_size, screen_size))
            pygame.display.set_caption("AddictionEnv Game")
            self.font = pygame.font.Font(None, 36)

        colors = {
            0: (255, 255, 255), # WHITE
            1: (0, 0, 255), # BLUE
            2: (255, 0, 0), # RED
            3: (0, 255, 0) # GREEN
        }

        for row in range(self.size):
            for col in range(self.size):
                color = colors[self.state[row, col]]
                pygame.draw.rect(self.screen, color, pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size), 1)

        # Create texts
        episodes_text = self.font.render(f'Episodes: [{episode}/{num_episodes}]', True, (20, 20, 20))
        current_reward_text = self.font.render(f'Reward: {reward:.2f}', True, (20, 20, 20))
        health_text = self.font.render(f'Health: {health:.2f}', True, (20, 20, 20))
        eps_text = self.font.render(f'Epsilon: {eps:.5f}', True, (20, 20, 20))

        # Render the texts
        self.screen.blit(episodes_text, (0, 0))
        self.screen.blit(current_reward_text, (0, 20))
        self.screen.blit(health_text, (0, 40))
        self.screen.blit(eps_text, (0, 60))

        pygame.display.flip() # Update frame

        # Quit the env
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def close(self):
        # Close the env
        if self.screen is not None:
            pygame.quit()
            self.screen = None
