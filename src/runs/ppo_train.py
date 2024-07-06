import numpy as np
from src.agents.ppo_agent import PPOAgent
from src.utils import shape_rewards, ParallelSoccerEnv
from tqdm import tqdm
from src.logger import CustomLogger
from src.config import N_GAMES


def train_ppo(n_games, n_agents, batch_size=64, n_envs=4):
    env = ParallelSoccerEnv(n_envs)
    ppo_agent = PPOAgent(336, 3)  # Single agent shared across all environments
    logger = CustomLogger("ppo", run_name="ppo_v3")

    for i in tqdm(range(n_games)):
        obs = env.reset()
        done = np.array([False] * n_envs)
        scores = [{j: 0 for j in range(4)} for _ in range(n_envs)]
        episode_states = []
        episode_actions = []
        episode_probs = []
        episode_rewards = []
        episode_dones = []

        while not done.all():
            env_actions = [{j: [0, 0, 0] for j in range(4)} for _ in range(n_envs)]
            for env_idx in range(n_envs):
                for j in range(4):
                    if j < n_agents:
                        actions, action_probs = ppo_agent.act(obs[env_idx][j])
                        env_actions[env_idx][j] = actions
                        episode_states.append(obs[env_idx][j])
                        episode_actions.append(actions)
                        episode_probs.append(action_probs)

            next_obs, rewards, done, info = env.step(env_actions)

            for env_idx in range(n_envs):
                for agent_id in range(4):
                    shaped_reward = shape_rewards(info[env_idx], int(agent_id))
                    scores[env_idx][agent_id] = shaped_reward
                    if agent_id < n_agents:
                        episode_rewards.append(shaped_reward)
                        episode_dones.append(done[env_idx]["__all__"])

            obs = next_obs

            # Perform update if batch size is reached
            if len(episode_states) >= batch_size:

                ppo_agent.update(
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
            ppo_agent.update(
                episode_states,
                episode_actions,
                episode_probs,
                episode_rewards,
                episode_dones,
            )

        logger.write_logs_and_tensorboard(
            i,
            scores[0],
            next_obs[0],
            rewards[0],
            done[0],
            info[0],
            env_actions[0],
            ppo_agent,
        )

    env.close()
    return ppo_agent


if __name__ == "__main__":
    trained_agent = train_ppo(n_games=N_GAMES, n_agents=1, batch_size=1024, n_envs=12)
