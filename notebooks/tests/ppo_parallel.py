from src.utils import shape_rewards, ParallelSoccerEnv
from tqdm import tqdm
from src.config import N_GAMES
from src.agents.ppo_agent import PPOAgent
from src.logger import CustomLogger
import numpy as np


def train_ppo(n_games, n_agents=1, batch_size=128, n_envs=4):
    env = ParallelSoccerEnv(n_envs)
    ppo_agent = PPOAgent(336, 3)  # Assuming state size is 336 and action size is 3
    logger = CustomLogger("ppo", run_name="ppo_v1")

    for episode in tqdm(range(n_games)):
        obs = env.reset()
        done = np.array([False] * n_envs)
        scores = [{j: 0 for j in range(4)} for _ in range(n_envs)]
        episode_steps = 0

        states, actions, old_probs, rewards, dones = [], [], [], [], []

        while not done.all():
            env_actions = [{j: [0, 0, 0] for j in range(4)} for _ in range(n_envs)]
            env_probs = [{j: [0, 0, 0] for j in range(4)} for _ in range(n_envs)]

            for env_idx in range(n_envs):
                episode_steps += 1

                for i in range(4):
                    if i < n_agents:
                        action, prob = ppo_agent.act(obs[env_idx][i])
                        env_actions[env_idx][i] = action
                        env_probs[env_idx][i] = prob

            next_obs, reward, done, info = env.step(env_actions)

            for env_idx in range(n_envs):
                for i in range(4):
                    shaped_reward = shape_rewards(info[env_idx], int(i))
                    scores[env_idx][i] = shaped_reward

                    if i < n_agents:
                        states.append(obs[env_idx][i])
                        actions.append(env_actions[env_idx][i])
                        old_probs.append(env_probs[env_idx][i])
                        rewards.append(shaped_reward)
                        dones.append(done[env_idx]["__all__"])

            obs = next_obs

            # Perform PPO update if we have enough samples
            if len(states) >= batch_size:
                ppo_agent.update(states, actions, old_probs, rewards, dones)
                states, actions, old_probs, rewards, dones = [], [], [], [], []

        # Perform final update with remaining samples
        if states:
            ppo_agent.update(states, actions, old_probs, rewards, dones)

        # Logging
        logger.write_logs_and_tensorboard(
            episode,
            scores[0],
            next_obs[0],
            reward[0],
            done[0],
            info[0],
            env_actions[0],
            ppo_agent,
            custom={"steps": episode_steps},
        )

    env.close()
    return ppo_agent


if __name__ == "__main__":
    trained_agent = train_ppo(n_games=N_GAMES, n_agents=1, batch_size=4)
