import random
from tqdm import tqdm
import soccer_twos
from src.agents.sac_agent import SACAgent
from src.utils import shape_rewards
from src.config import N_GAMES
from src.logger import CustomLogger
import numpy as np


def train_sac(n_games, n_agents, batch_size=64):
    env = soccer_twos.make()
    sac_agents = SACAgent(336, 3)
    logger = CustomLogger("sac", run_name="sac_v3_single")

    for i in tqdm(range(n_games)):
        obs = env.reset()
        done = False
        scores = {j: 0 for j in range(4)}
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_next_states = []
        episode_dones = []

        while not done:
            actions = {}
            for j in range(4):
                actions[j] = [0, 0, 0]
                if j < n_agents:
                    actions[j] = sac_agents.act(obs[j])

            next_obs, reward, done, info = env.step(actions)
            done = done["__all__"]

            for agent_id in range(4):
                scores[agent_id] = reward[agent_id] + shape_rewards(info, int(agent_id))

            # Only store data for the SAC agent (agent 0)
            episode_states.append(obs[0])
            episode_actions.append(actions[0])
            episode_rewards.append(scores[0])
            episode_next_states.append(next_obs[0])
            episode_dones.append(done)

            obs = next_obs

            # Update the SAC agent if we have enough samples
            if len(episode_states) >= batch_size:
                sac_agents.update(
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
            sac_agents.update(
                np.array(episode_states),
                np.array(episode_actions),
                np.array(episode_rewards),
                np.array(episode_next_states),
                np.array(episode_dones),
            )

        logger.write_logs_and_tensorboard(
            i, scores, next_obs, reward, done, info, actions, sac_agents
        )

    env.close()


if __name__ == "__main__":
    train_sac(n_games=N_GAMES, n_agents=1, batch_size=128)
