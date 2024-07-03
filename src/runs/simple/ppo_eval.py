import random
from src.utils import shape_rewards
from tqdm import tqdm
import soccer_twos
from src.config import N_GAMES
from src.agents.ppo_agent import PPOAgents
from src.logger import CustomLogger
import numpy as np


def train_ppo(n_games, n_agents):
    env = soccer_twos.make(worker_id=random.randint(0, 100))

    agent_indices = list(range(n_agents))

    ppo_agents = PPOAgents(n_agents, 336, 3)

    logger = CustomLogger("ppo")

    for i in tqdm(range(n_games)):
        obs = env.reset()
        done = False
        scores = {}
        episode_states = {agent_id: [] for agent_id in agent_indices}
        episode_actions = {agent_id: [] for agent_id in agent_indices}
        episode_probs = {agent_id: [] for agent_id in agent_indices}
        episode_rewards = {agent_id: [] for agent_id in agent_indices}
        episode_dones = []

        while not done:
            actions, action_probs = ppo_agents.act({i: obs[i] for i in agent_indices})
            env_actions = {}
            for j in range(4):
                if j < len(agent_indices):
                    env_actions[j] = actions[j]
                else:
                    env_actions[j] = [0, 0, 0]  # Default action for non-agent players

            next_obs, reward, done, info = env.step(env_actions)
            done = done["__all__"]

            for agent_id in range(4):
                scores[agent_id] = reward[agent_id] + shape_rewards(info, int(agent_id))

            for agent_id in agent_indices:
                episode_states[agent_id].append(obs[agent_id])
                episode_actions[agent_id].append(actions[agent_id])
                episode_probs[agent_id].append(
                    action_probs[agent_id]
                )  # This now stores only the probability of the chosen action
                episode_rewards[agent_id].append(scores[agent_id])
            episode_dones.append(done)

            obs = next_obs

        ppo_agents.update(
            episode_states,
            episode_actions,
            episode_probs,
            episode_rewards,
            episode_dones,
        )

        logger.write_logs_and_tensorboard(
            i, scores, next_obs, reward, done, info, actions, ppo_agents
        )

    env.close()


if __name__ == "__main__":
    train_ppo(n_games=N_GAMES, n_agents=1)
