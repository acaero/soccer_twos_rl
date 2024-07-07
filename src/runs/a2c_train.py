import random
from tqdm import tqdm
import soccer_twos
from src.agents.a2c_agent import A2CAgent
from src.utils import shape_rewards
from src.config import N_GAMES
from src.logger import CustomLogger
import numpy as np


def train_a2c(n_games, n_agents, update_frequency=5):
    env = soccer_twos.make(worker_id=4)
    a2c_agent = A2CAgent(336, 3)
    logger = CustomLogger("a2c", run_name="a2c_v1_single")

    for i in tqdm(range(n_games)):
        obs = env.reset()
        done = False
        scores = {j: 0 for j in range(4)}
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_next_states = []
        episode_dones = []
        step_count = 0

        while not done:
            actions = {}
            for j in range(4):
                actions[j] = [0, 0, 0]
                if j < n_agents:
                    actions[j] = a2c_agent.act(obs[j])

            next_obs, reward, done, info = env.step(actions)
            done = done["__all__"]

            for agent_id in range(4):
                scores[agent_id] = shape_rewards(info, int(agent_id))

            # Only store data for the A2C agent (agent 0)
            episode_states.append(obs[0])
            episode_actions.append(actions[0])
            episode_rewards.append(scores[0])
            episode_next_states.append(next_obs[0])
            episode_dones.append(done)

            obs = next_obs
            step_count += 1

            # Update the A2C agent every update_frequency steps
            if step_count % update_frequency == 0:
                a2c_agent.update(
                    np.array(episode_states),
                    np.array(episode_actions),
                    np.array(episode_rewards),
                    np.array(episode_next_states),
                    np.array(episode_dones),
                )
                episode_states = []
                episode_actions = []
                episode_rewards = []
                episode_next_states = []
                episode_dones = []

        # Perform final update with remaining experiences
        if episode_states:
            a2c_agent.update(
                np.array(episode_states),
                np.array(episode_actions),
                np.array(episode_rewards),
                np.array(episode_next_states),
                np.array(episode_dones),
            )

        logger.write_logs_and_tensorboard(
            i, scores, next_obs, reward, done, info, actions, a2c_agent
        )

    env.close()


if __name__ == "__main__":
    train_a2c(n_games=N_GAMES, n_agents=1, update_frequency=4)
