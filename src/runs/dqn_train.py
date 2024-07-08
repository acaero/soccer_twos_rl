import random
from src.utils import shape_rewards
from tqdm import tqdm
import soccer_twos
from src.config import N_GAMES
from src.agents.dqn_agent import DQNAgent
from src.logger import CustomLogger


def train_dqn(n_games):
    env = soccer_twos.make()
    dqn_agent = DQNAgent(336, 3)
    logger = CustomLogger("dqn", run_name="dqn_v1")

    for i in tqdm(range(n_games)):
        print(f"Starting episode {i} with epsilon: {dqn_agent.epsilon}")

        obs = env.reset()
        done = False
        scores = {0: 0, 1: 0, 2: 0, 3: 0}
        episode_loss = 0
        step_count = 0

        while not done:
            actions = {}
            for j in range(4):
                actions[j] = dqn_agent.act(obs[j]) if j == 0 else [0, 0, 0]

            next_obs, reward, done, info = env.step(actions)
            done = done["__all__"]

            for agent_id in range(4):
                shaped_reward = shape_rewards(info, int(agent_id))
                scores[agent_id] = shaped_reward

            dqn_agent.remember(obs[0], actions[0], shaped_reward, next_obs[0], done)
            loss = dqn_agent.replay()
            if loss is not None:
                episode_loss += loss

            obs = next_obs
            step_count += 1

        # Decay epsilon after each episode
        dqn_agent.epsilon = max(
            dqn_agent.epsilon_min, dqn_agent.epsilon * dqn_agent.epsilon_decay
        )

        avg_loss = episode_loss / step_count if step_count > 0 else 0

        logger.write_logs_and_tensorboard(
            i,
            scores,
            next_obs,
            reward,
            done,
            info,
            actions,
            dqn_agent,
            custom={
                "epsilon": dqn_agent.epsilon,
                "avg_loss": avg_loss,
                "steps": step_count,
            },
        )

    env.close()


if __name__ == "__main__":
    train_dqn(n_games=N_GAMES)
