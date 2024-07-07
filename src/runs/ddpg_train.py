import random
from src.utils import shape_rewards
from tqdm import tqdm
import soccer_twos
from src.config import N_GAMES
from src.agents.ddpg_agent import DDPGAgent
from src.logger import CustomLogger
import numpy as np


def train_ddpg(n_games, n_agents, update_every=4):
    env = soccer_twos.make(worker_id=3)
    ddpg_agent = DDPGAgent(336, 3)  # Assuming state size is 336 and action size is 3
    logger = CustomLogger("ddpg", run_name="ddpg_v1")

    total_steps = 0

    for i in tqdm(range(n_games)):
        obs = env.reset()
        done = False
        scores = {0: 0, 1: 0, 2: 0, 3: 0}
        episode_steps = 0

        while not done:
            actions = {}
            for j in range(4):
                if j < n_agents:
                    actions[j] = ddpg_agent.act(obs[j])
                else:
                    actions[j] = [0, 0, 0]

            next_obs, reward, done, info = env.step(actions)
            done = done["__all__"]

            for agent_id in range(4):
                shaped_reward = shape_rewards(info, int(agent_id))
                scores[agent_id] = shaped_reward

                if agent_id < n_agents:
                    ddpg_agent.remember(
                        obs[agent_id],
                        actions[agent_id],
                        shaped_reward,
                        next_obs[agent_id],
                        done,
                    )

            obs = next_obs
            episode_steps += 1
            total_steps += 1

            # Perform update every few steps
            if total_steps % update_every == 0:
                ddpg_agent.update()

        # Logging
        avg_score = np.mean([scores[j] for j in range(n_agents)])
        logger.write_logs_and_tensorboard(
            i,
            scores,
            next_obs,
            reward,
            done,
            info,
            actions,
            ddpg_agent,
            custom={"avg_score": avg_score, "episode_steps": episode_steps},
        )

    env.close()
    return ddpg_agent


if __name__ == "__main__":
    train_ddpg(n_games=N_GAMES, n_agents=1, update_every=4)
