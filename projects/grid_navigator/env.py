import pygame
import numpy as np


class GridNavigatorEnv:
    def __init__(self, size, cell_size = 40, init_state = False):
        self.size = size
        self.screen = None # Pygame screen object, initialized later
        self.cell_size = cell_size # Size of each cell in the grid in pixels
        self.state = None
        self.done = None

        if init_state: self.reset()

    def reset(self):
        # Randomly place the agent and the point in the grid
        self.agent_pos = np.random.randint(0, self.size, 2)
        self.point_pos = np.random.randint(0, self.size, 2)

        self.state = self._get_state()
        self.done = False

        return self.state

    def _get_state(self):
        state = np.zeros((1, self.size, self.size), dtype=np.float32)
        state[0, self.agent_pos[0], self.agent_pos[1]] = 1 # Agent
        state[0, self.agent_pos[0], self.agent_pos[1]] = 10 # Point

        return state # Return the state as a 2D grid

    def step(self, action):
        if action == 0:
            self.agent_pos[0] -= 1  # Up
        elif action == 1:
            self.agent_pos[0] += 1  # Down
        elif action == 2:
            self.agent_pos[1] -= 1  # Left
        elif action == 3:
            self.agent_pos[1] += 1  # Right

        # Ensure the agent stays within the grid boundaries
        self.agent_pos = np.clip(self.agent_pos, 0, self.size - 1)

        reward = self._reward_calculation()

        return self._get_state(), reward, self.done # Return the new state, reward, and done flag

    def _reward_calculation(self):
        reward = -0.1 # Default reward for each step

        if np.array_equal(self.agent_pos, self.point_pos):
            reward = 15 # Reward for finding the point
            self.done = True

        return reward

    def _is_screen_none(self):
        # Initialize the Pygame screen if not already done
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.size * self.cell_size, self.size * self.cell_size))
            pygame.display.set_caption("DCQN Agent in 2D Grid World")


    def render(self):
        self._is_screen_none()

        # Fill the screen with a white background
        self.screen.fill((255, 255, 255))

        # Draw the grid with the agent and food
        for row in range(self.size):
            for col in range(self.size):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                if np.array_equal(self.agent_pos, [row, col]):
                    pygame.draw.rect(self.screen, (0, 0, 255), rect)  # Blue for the agent
                elif np.array_equal(self.food_pos, [row, col]):
                    pygame.draw.rect(self.screen, (255, 0, 0), rect)  # Red for the food
                else:
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)  # Grey grid lines

        pygame.display.flip()  # Update the display

    def close(self):
        # Close the Pygame window if it was opened
        if self.screen:
            pygame.quit()
