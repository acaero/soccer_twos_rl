import random
from tqdm import tqdm
import soccer_twos
from src.agents.sac_agents import SACAgents
from src.utils import shape_rewards
from src.config import N_GAMES
from src.logger import CustomLogger
import numpy as np


def train_sac(n_games, n_agents):
    env = soccer_twos.make(worker_id=random.randint(0, 100))

    agent_indices = list(range(n_agents))

    sac_agents = SACAgents(n_agents, 336, 3)

    logger = CustomLogger("sac")

    for i in tqdm(range(n_games)):
        obs = env.reset()
        done = False
        scores = {}
        episode_states = {agent_id: [] for agent_id in agent_indices}
        episode_actions = {agent_id: [] for agent_id in agent_indices}
        episode_rewards = {agent_id: [] for agent_id in agent_indices}
        episode_next_states = {agent_id: [] for agent_id in agent_indices}
        episode_dones = []

        while not done:
            actions = sac_agents.act({i: obs[i] for i in agent_indices})

            # Ensure all 4 agents have actions
            for j in range(4):
                if j not in actions:
                    actions[j] = [0, 0, 0]  # Default action for non-agent players

            next_obs, reward, done, info = env.step(actions)
            done = done["__all__"]

            for agent_id in range(4):
                scores[agent_id] = reward[agent_id] + shape_rewards(info, int(agent_id))

            for agent_id in agent_indices:
                episode_states[agent_id] = np.array(episode_states[agent_id])
                episode_actions[agent_id] = np.array(episode_actions[agent_id])
                episode_rewards[agent_id] = np.array(episode_rewards[agent_id])
                episode_next_states[agent_id] = np.array(episode_next_states[agent_id])
            episode_dones = np.array(episode_dones)

            sac_agents.update(
                episode_states,
                episode_actions,
                episode_rewards,
                episode_next_states,
                episode_dones,
            )

            obs = next_obs

        logger.write_logs_and_tensorboard(
            i, scores, next_obs, reward, done, info, actions, sac_agents
        )

    env.close()


if __name__ == "__main__":
    train_sac(n_games=N_GAMES, n_agents=1)
