import random
from src.utils import shape_rewards
from tqdm import tqdm
import soccer_twos
from src.config import N_GAMES
from src.agents.ddqn_agent import DDQNAgent
from src.logger import CustomLogger


def train_ddqn(n_games, n_agents):
    env = soccer_twos.make()

    ddqn_agent = DDQNAgent(336, 3)

    logger = CustomLogger("ddqn", run_name="ddqn_v1")

    for i in tqdm(range(n_games)):
        obs = env.reset()
        done = False
        scores = {}
        while not done:

            actions = {}
            for j in range(4):
                actions[j] = [0, 0, 0]
                if j < n_agents:
                    actions[j] = ddqn_agent.act(obs[j])

            next_obs, reward, done, info = env.step(actions)
            done = done["__all__"]

            for agent_id in range(4):
                scores[agent_id] = reward[agent_id] + shape_rewards(info, int(agent_id))

            ddqn_agent.remember(obs[0], actions[0], scores[0], next_obs[0], done)
            ddqn_agent.replay()

            obs = next_obs

        logger.write_logs_and_tensorboard(
            i,
            scores,
            next_obs,
            reward,
            done,
            info,
            actions,
            ddqn_agent,
            custom={"epsilon": ddqn_agent.epsilon},
        )

    env.close()


if __name__ == "__main__":
    train_ddqn(n_games=N_GAMES, n_agents=1)
