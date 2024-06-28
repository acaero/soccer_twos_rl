from src.utils import shape_rewards
from tqdm import tqdm
import soccer_twos
from src.config import N_GAMES
from src.agents.ddqn_agent import DDQNAgents
from src.logger import CustomLogger


def train_ddqn(n_games, n_agents):
    env = soccer_twos.make()

    agent_indices = []
    for i in range(n_agents):
        agent_indices.append(i)

    ddqn_agents = DDQNAgents(
        n_agents, 336, 3, buffer_size=10000, batch_size=1024, learning_rate=0.001
    )

    logger = CustomLogger("ddqn")

    for i in tqdm(range(n_games)):
        obs = env.reset()
        done = False
        scores = {}
        while not done:

            actions = ddqn_agents.act({i: obs[i] for i in agent_indices})
            for j in range(len(agent_indices), 4):
                actions[j] = [0, 0, 0]

            next_obs, reward, done, info = env.step(actions)
            done = done["__all__"]

            for agent_id in range(4):
                scores[agent_id] = reward[agent_id] + shape_rewards(info, int(agent_id))

            ddqn_agents.remember(obs, actions, scores, next_obs, done)
            ddqn_agents.replay()

            obs = next_obs

        logger.write_logs_and_tensorboard(
            i,
            scores,
            next_obs,
            reward,
            done,
            info,
            ddqn_agents,
            custom={"epsilon": ddqn_agents.epsilon},
        )

    env.close()


if __name__ == "__main__":
    train_ddqn(n_games=N_GAMES, n_agents=2)
