from src.utils import shape_rewards
from src.config import N_GAMES
import soccer_twos
from src.logger import CustomLogger
from tqdm import tqdm
from src.agents.maddpg_agent import MADDPGAgents


def train_maddpg(
    n_games=N_GAMES,
    n_agents=1,
):
    env = soccer_twos.make()

    agent_indices = []
    for i in range(n_agents):
        agent_indices.append(i)

    maddpg_agent = MADDPGAgents(n_agents, 336, 3)
    logger = CustomLogger("maddpg", run_name="maddpg_v1")
    # Collect initial experiences
    initial_experiences = 1000
    experiences_count = 0
    obs = env.reset()

    while experiences_count < initial_experiences:
        scores = {}
        actions = maddpg_agent.act({i: obs[i] for i in agent_indices})
        for j in range(len(agent_indices), 4):
            actions[j] = [0, 0, 0]

        next_obs, reward, dones, info = env.step(actions)
        for agent_id in agent_indices:
            scores[agent_id] = reward[agent_id] + shape_rewards(info, int(agent_id))
        done = dones["__all__"]
        maddpg_agent.remember(
            {j: obs[j] for j in agent_indices},
            {j: actions[j] for j in agent_indices},
            {j: scores[j] for j in agent_indices},
            {j: next_obs[j] for j in agent_indices},
            dones,
        )
        experiences_count += 1
        obs = next_obs
        if done:
            obs = env.reset()

    print(f"Collected {experiences_count} initial experiences")

    for i in tqdm(range(n_games)):
        obs = env.reset()
        done = False
        scores = {}
        while not done:
            actions = maddpg_agent.act({i: obs[i] for i in agent_indices})
            for j in range(len(agent_indices), 4):
                actions[j] = [0, 0, 0]

            next_obs, reward, dones, info = env.step(actions)
            done = dones["__all__"]

            for agent_id in range(4):
                scores[agent_id] = reward[agent_id] + shape_rewards(info, int(agent_id))

            maddpg_agent.remember(
                {j: obs[j] for j in agent_indices},
                {j: actions[j] for j in agent_indices},
                {j: scores[j] for j in agent_indices},
                {j: next_obs[j] for j in agent_indices},
                dones,
            )

            maddpg_agent.replay()

            obs = next_obs

        logger.write_logs_and_tensorboard(
            i, scores, next_obs, reward, done, info, actions, maddpg_agent
        )

    env.close()


if __name__ == "__main__":
    train_maddpg(n_games=N_GAMES, n_agents=1)
