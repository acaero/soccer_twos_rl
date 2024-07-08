import random
from src.utils import shape_rewards
from tqdm import tqdm
import soccer_twos
from src.config import N_GAMES
from src.agents.ppo_agent import PPOAgent
from src.logger import CustomLogger
import numpy as np


def train_ppo(n_games, n_agents=1, batch_size=128):
    env = soccer_twos.make(worker_id=2)
    ppo_agent = PPOAgent(336, 3)  # Assuming state size is 336 and action size is 3
    logger = CustomLogger("ppo", run_name="ppo_clear_v2")

    for episode in tqdm(range(n_games)):
        obs = env.reset()
        done = False
        episode_rewards = {i: 0 for i in range(4)}
        episode_steps = 0

        states, actions, old_probs, rewards, dones = [], [], [], [], []

        while not done:
            episode_steps += 1
            agent_actions = {}
            agent_probs = {}

            for i in range(4):
                if i < n_agents:
                    action, prob = ppo_agent.act(obs[i])
                    agent_actions[i] = action
                    agent_probs[i] = prob
                else:
                    agent_actions[i] = [0, 0, 0]

            next_obs, reward, done, info = env.step(agent_actions)
            done = done["__all__"]

            for i in range(4):
                shaped_reward = shape_rewards(info, i)
                episode_rewards[i] = shaped_reward

                if i < n_agents:
                    states.append(obs[i])
                    actions.append(agent_actions[i])
                    old_probs.append(agent_probs[i])
                    rewards.append(shaped_reward)
                    dones.append(done)

            obs = next_obs

            # Perform PPO update if we have enough samples
            if len(states) >= batch_size:
                ppo_agent.update(states, actions, old_probs, rewards, dones)
                states, actions, old_probs, rewards, dones = [], [], [], [], []

        # Perform final update with remaining samples
        if states:
            ppo_agent.update(states, actions, old_probs, rewards, dones)

        # Logging
        avg_reward = np.mean([episode_rewards[i] for i in range(n_agents)])
        logger.write_logs_and_tensorboard(
            episode,
            episode_rewards,
            next_obs,
            reward,
            done,
            info,
            agent_actions,
            ppo_agent,
            custom={"avg_reward": avg_reward, "steps": episode_steps},
        )

    env.close()
    return ppo_agent


if __name__ == "__main__":
    trained_agent = train_ppo(n_games=N_GAMES, n_agents=1, batch_size=4)
