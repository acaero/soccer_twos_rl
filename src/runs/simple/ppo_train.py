import random
from src.utils import shape_rewards
from tqdm import tqdm
import soccer_twos
from src.config import N_GAMES
from src.agents.ppo_agent import PPOAgent
from src.logger import CustomLogger
import numpy as np
import torch


def train_ppo(n_games, n_agents, batch_size=64):
    env = soccer_twos.make()
    ppo_agents = PPOAgent(336, 3)
    logger = CustomLogger("ppo")

    for i in tqdm(range(n_games)):
        obs = env.reset()
        done = False
        scores = {j: 0 for j in range(4)}
        episode_states = []
        episode_actions = []
        episode_probs = []
        episode_rewards = []
        episode_dones = []

        while not done:
            env_actions = {}
            for j in range(4):
                env_actions[j] = [0, 0, 0]
                if j < n_agents:
                    actions, action_probs = ppo_agents.act(obs[j])
                    env_actions[j] = actions

                    episode_states.append(obs[j])
                    episode_actions.append(actions)
                    episode_probs.append(action_probs)

            next_obs, reward, done, info = env.step(env_actions)
            done = done["__all__"]

            for agent_id in range(4):
                shaped_reward = shape_rewards(info, int(agent_id))
                scores[agent_id] = shaped_reward
                if agent_id < n_agents:
                    episode_rewards.append(shaped_reward)
                    episode_dones.append(done)
            print(scores[0])
            obs = next_obs

            # Perform update if batch size is reached
            if len(episode_states) >= batch_size:
                ppo_agents.update(
                    episode_states,
                    episode_actions,
                    episode_probs,
                    episode_rewards,
                    episode_dones,
                )
                episode_states = []
                episode_actions = []
                episode_probs = []
                episode_rewards = []
                episode_dones = []

        # Perform final update with remaining experiences
        if episode_states:
            ppo_agents.update(
                episode_states,
                episode_actions,
                episode_probs,
                episode_rewards,
                episode_dones,
            )

        logger.write_logs_and_tensorboard(
            i, scores, next_obs, reward, done, info, env_actions, ppo_agents
        )

    env.close()


if __name__ == "__main__":
    train_ppo(n_games=N_GAMES, n_agents=1, batch_size=4)
