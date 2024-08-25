from env import AddictionEnv
from agent import Agent
import logging


class Trainer:
    def __init__(
            self,
            num_episodes = 1000,
            max_steps_per_episode = 200,
            update_target_every = 10,
            buffer_size = 10000,
            batch_size = 64,
            gamma = 0.99,
            lr = 0.001,
            epsilon = 1.0,
            epsilon_decay = 0.995,
            epsilon_min = 0.1,
            show_logs = False
        ):

        self.episodes = num_episodes
        self.max_steps = max_steps_per_episode
        self.update_target_every = update_target_every

        self.env = AddictionEnv()
        state_size = self.env.size ** 2
        action_size = 4

        self.show_logs = show_logs

        self.agent = Agent(
            state_size=state_size,
            action_size=action_size,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            lr=lr,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min
        )

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('DQNTrainer')

    def train_episode(self, episode):
        state = self.env.reset().flatten()
        total_reward = 0

        for _ in range(self.max_steps):
            action = self.agent.act(state)
            next_state, reward, done, info = self.env.step(action)
            health = info['health']
            next_state = next_state.flatten()
            self.agent.remember(state, action, reward, next_state, done, health)
            state = next_state
            total_reward += reward

            if done:
                break

            self.agent.replay()

            self.env.render(episode, self.episodes, total_reward, health, self.agent.epsilon)

        return total_reward, health

    def learn(self):
        for e in range(self.episodes):
            total_reward, health = self.train_episode(e + 1)
            if self.show_logs: self.logger.info(f"Episode {e + 1}/{self.episodes}, Reward: {total_reward:.2f}, Epsilon: {self.agent.epsilon:.2f}, Health: {health}")

            if (e + 1) % self.update_target_every == 0:
                self.agent.update_target_model()

        self.env.close()