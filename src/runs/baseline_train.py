import random
from src.utils import shape_rewards
from tqdm import tqdm
import soccer_twos
from src.agents.baseline_agent import BaselineAgent
from src.agents.baseline_agent import RandomAgent
from src.logger import CustomLogger
from src.config import N_GAMES


def train_baseline(n_games, n_agents):
    env = soccer_twos.make(render=True)
    agent = BaselineAgent(n_agents)
    logger = CustomLogger("baseline", run_name="baseline_v1")
    actions = {
        0: [0, 0, 0],
        1: [0, 0, 0],
        2: [0, 0, 0],
        3: [0, 0, 0],
    }
    _, reward, done, info = env.step(actions)
    for i in tqdm(range(n_games)):
        obs = env.reset()
        done = False
        scores = {0: 0, 1: 0, 2: 0, 3: 0}
        while not done:
            actions = {
                player_id: agent.act(info, player_id) for player_id in range(n_agents)
            }
            for j in range(n_agents, 4):
                actions[j] = [0, 0, 0]
            next_obs, reward, done, info = env.step(actions)
            done = done["__all__"]
            for player_id in range(4):
                scores[player_id] = reward[player_id] + shape_rewards(info, player_id)
            obs = next_obs
        logger.write_logs_and_tensorboard(
            i, scores, next_obs, reward, done, info, actions, agent
        )
    env.close()


def train_random(n_games, n_agents):
    env = soccer_twos.make()
    agent = RandomAgent(n_agents)
    logger = CustomLogger("random")
    for i in tqdm(range(n_games)):
        obs = env.reset()
        done = False
        scores = {0: 0, 1: 0, 2: 0, 3: 0}
        while not done:
            actions = {
                player_id: agent.act(obs[player_id])[0] for player_id in range(n_agents)
            }
            for j in range(n_agents, 4):
                actions[j] = [0, 0, 0]
            next_obs, reward, done, info = env.step(actions)
            done = done["__all__"]
            for player_id in range(4):
                scores[player_id] = reward[player_id] + shape_rewards(info, player_id)
            obs = next_obs
        logger.write_logs_and_tensorboard(
            i, scores, next_obs, reward, done, info, actions, agent
        )
    env.close()


if __name__ == "__main__":
    train_baseline(n_games=N_GAMES, n_agents=1)
    # train_random(n_games=N_GAMES, n_agents=1)
