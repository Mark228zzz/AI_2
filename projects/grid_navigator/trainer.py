import torch
from env import GridNavigatorEnv
from agent import Agent


class Trainer:
    def __init__(
        self,
        device = None,
        map_size = 5,
        num_episodes = 1000,
        max_steps = 100,
        lr = 0.001,
        gamma = 0.99,
        epsilon = 1.0,
        epsilon_decay = 0.995,
        epsilon_min = 0.05,
        batch_size = 32,
        memory_size = 10000,
        tau = 0.001,
        cell_size = 40
    ):

        # Initialize parameters for the environment and training
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.map_size = map_size  # Size of the grid map
        self.num_episodes = num_episodes  # Total number of episodes to train
        self.max_steps = max_steps  # Maximum steps per episode
        self.batch_size = batch_size  # Batch size for experience replay
        self.cell_size = cell_size  # Size of each cell in the grid (for rendering)

        # Initialize environment and agent
        self.env = GridNavigatorEnv(self.map_size, self.cell_size)
        # Initialize the agent with the specified hyperparameters
        self.agent = Agent(
            state_shape=(1, self.map_size, self.map_size),
            action_size=4,  # Number of possible actions (Up, Down, Left, Right)
            gamma=gamma,  # Discount factor for future rewards
            lr=lr,  # Learning rate for the optimizer
            epsilon=epsilon,  # Initial exploration rate
            epsilon_decay=epsilon_decay,  # Rate at which exploration decreases
            epsilon_min=epsilon_min,  # Minimum exploration rate
            memory_size=memory_size,  # Size of the replay memory
            tau=tau  # Parameter for soft update of the target network
        )

    def train(self):
        # Main training loop over the episodes
        for episode in range(self.num_episodes):
            state = self.env.reset()  # Reset the environment to start a new episode
            total_reward = 0  # Initialize total reward for this episode

            for step in range(self.max_steps):
                action = self.agent.act(state)  # Agent selects an action
                next_state, reward, done = self.env.step(action)  # Environment responds to the action
                self.agent.remember(state, action, reward, next_state, done)  # Store the experience in replay memory
                state = next_state  # Update the current state
                total_reward += reward  # Accumulate the reward

                self.env.render(episode + 1, self.num_episodes, step, self.max_steps, total_reward, self.agent.epsilon)  # Render the current state of the environment

                if done:
                    self.agent.soft_update_target_model()  # Soft update the target network after the episode ends
                    break  # Exit the loop if the episode is finished

                self.agent.replay(batch_size=self.batch_size)  # Train the agent with experience replay

            print(f"Episode: [{episode + 1}/{self.num_episodes}], Total Reward: {total_reward}")

        self.env.close()
