from src.utils import convert_arrays_to_lists, RewardShaper
from src.config import CHECKPOINT_DIR, N_GAMES, IMAGES_DIR
import numpy as np
import soccer_twos
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.logger import CustomLogger
from pathlib import Path
from src.agents.ddpg_agent import DDPGAgent


def train_ddpg(n_games=N_GAMES, checkpoint_dir=CHECKPOINT_DIR):
    env = soccer_twos.make()
    logger = CustomLogger().logger
    reward_shaper = RewardShaper()

    scores, avg_scores = [], []

    ddpg_agent = DDPGAgent(336, 3)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(n_games)):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            actions = {}
            for player_id in range(4):
                if player_id == 0:
                    actions[player_id] = ddpg_agent.act(obs[player_id])
                else:
                    actions[player_id] = [0, 0, 0]  # Static action for other agents

            next_obs, reward, done, info = env.step(actions)
            done = done["__all__"]

            # Calculate the custom reward for the DDPG agent (player 0)
            reward = reward[0] + reward_shaper.calculate_reward(
                obs[0], next_obs[0], info, 0
            )

            ddpg_agent.remember(obs[0], actions[0], reward, next_obs[0], done)
            ddpg_agent.replay()

            obs = next_obs
            score += reward

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        if i % 10 == 0:
            print(f"Episode: {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}")
            checkpoint_filename = checkpoint_dir / f"checkpoint_DDPG_{i}.pth"
            ddpg_agent.save(checkpoint_filename)

        logger.info(
            "DDPG Agent",
            extra={
                "custom_fields": {
                    "episode": i,
                    "score": score,
                    "average_score": avg_score,
                    "reward": str(convert_arrays_to_lists(reward)),
                    "done": str(convert_arrays_to_lists(done)),
                    "info": str(convert_arrays_to_lists(info)),
                    "observations": str(convert_arrays_to_lists(obs)),
                }
            },
        )

    return scores, avg_scores


def visualize_performance(episodes, scores, avg_scores, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    ax1.plot(episodes, scores, label="Score", alpha=0.6)
    ax1.plot(episodes, avg_scores, label="Average Score", linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Score")
    ax1.set_title("DDPG Performance")
    ax1.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    scores, avg_scores = train_ddpg(n_games=N_GAMES)
    episodes = list(range(1, len(scores) + 1))
    image_dir = Path(IMAGES_DIR)
    image_dir.mkdir(parents=True, exist_ok=True)
    visualize_performance(
        episodes,
        scores,
        avg_scores,
        image_dir / "soccer_twos_ddpg_performance.png",
    )
