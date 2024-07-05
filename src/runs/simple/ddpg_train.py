import random
from src.utils import shape_rewards
from tqdm import tqdm
import soccer_twos
from src.config import N_GAMES
from src.agents.ddpg_agent import DDPGAgent
from src.logger import CustomLogger


def train_ddpg(n_games, n_agents):
    env = soccer_twos.make(render=True)

    ddpg_agent = DDPGAgent(336, 3)

    logger = CustomLogger("ddpg")

    for i in tqdm(range(n_games)):
        obs = env.reset()
        done = False
        scores = {}
        while not done:

            actions = {}
            for j in range(4):
                actions[j] = [0, 0, 0]
                if j < n_agents:
                    actions[j] = ddpg_agent.act(obs[j])

            next_obs, reward, done, info = env.step(actions)
            done = done["__all__"]

            for agent_id in range(4):
                scores[agent_id] = reward[agent_id] + shape_rewards(info, int(agent_id))

            ddpg_agent.remember(obs[0], actions[0], scores[0], next_obs[0], done)
            ddpg_agent.replay()

            obs = next_obs

        logger.write_logs_and_tensorboard(
            i, scores, next_obs, reward, done, info, actions, ddpg_agent
        )

    env.close()


# if __name__ == "__main__":
#     train_ddpg(n_games=N_GAMES, n_agents=1)
