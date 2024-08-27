import sys
import pygame
import numpy as np



class GridNavigatorEnv:
    def __init__(self, size, cell_size = 40, init_state = False):
        pygame.init()

        self.size = size
        self.screen = None # Pygame screen object, initialized later
        self.cell_size = cell_size # Size of each cell in the grid in pixels
        self.state = None
        self.done = None
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 50)

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


    def render(self, episode, num_episodes, step, max_steps, reward, epsilon):
        self._is_screen_none()

        # Fill the screen with a white background
        self.screen.fill((255, 255, 255))

        # Draw the grid with the agent and point
        for row in range(self.size):
            for col in range(self.size):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                if np.array_equal(self.agent_pos, [row, col]):
                    pygame.draw.rect(self.screen, (0, 0, 255), rect)  # Blue for the agent
                elif np.array_equal(self.point_pos, [row, col]):
                    pygame.draw.rect(self.screen, (255, 0, 0), rect)  # Red for the point
                else:
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)  # Grey grid lines

        # Create texts surface objects
        text_episodes = self.font.render(f'Episodes: [{episode}/{num_episodes}]', True, (90, 90 ,90))
        text_steps = self.font.render(f'Steps: [{step}/{max_steps}]', True, (90, 90 ,90))
        text_reward = self.font.render(f'Reward: {reward:.2f}', True, (90, 90 ,90))
        text_epsilon = self.font.render(f'Epsilon: {epsilon:.3f}', True, (90, 90 ,90))

        # Draw the texts on the screen
        self.screen.blit(text_episodes, (0, 0))
        self.screen.blit(text_steps, (0, 30))
        self.screen.blit(text_reward, (0, 60))
        self.screen.blit(text_epsilon, (0, 90))

        pygame.display.flip()  # Update the display
        self.clock.tick(120)

        # Quit the env
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def close(self):
        # Close the Pygame window if it was opened
        if self.screen:
            pygame.quit()
