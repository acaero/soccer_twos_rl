from src.utils import convert_arrays_to_lists, RewardShaper
from src.config import CHECKPOINT_DIR, N_GAMES, IMAGES_DIR, LOG_DIR
import numpy as np
import soccer_twos
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.logger import CustomLogger
from pathlib import Path
from src.agents.ddpg_agent import DDPGAgent
from torch.utils.tensorboard import SummaryWriter


def train_ddpg(
    n_games=N_GAMES,
    checkpoint_dir=CHECKPOINT_DIR,
    log_dir=LOG_DIR,
    images_dir=IMAGES_DIR,
):
    env = soccer_twos.make()
    reward_shaper = RewardShaper()

    scores, avg_scores = [], []

    ddpg_agent = DDPGAgent(336, 3)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_log_dir = Path(log_dir) / "tensorboard"
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    plain_log_dir = Path(log_dir) / "plain"
    plain_log_dir.mkdir(parents=True, exist_ok=True)
    logger = CustomLogger(log_dir=plain_log_dir).logger

    for i in tqdm(range(n_games)):
        obs = env.reset()
        done = False
        score = 0
        episode_loss = 0
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
            loss = ddpg_agent.replay()
            if loss is not None:
                episode_loss += loss

            obs = next_obs
            score += reward

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        # Log metrics to TensorBoard
        writer.add_scalar("Score", score, i)
        writer.add_scalar("Average Score", avg_score, i)
        writer.add_scalar("Loss/Episode Loss", episode_loss, i)

        # Log metrics to plain log file
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

        if i % 10 == 0:
            print(f"Episode: {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}")
            checkpoint_filename = checkpoint_dir / f"checkpoint_DDPG_{i}.pth"
            ddpg_agent.save(checkpoint_filename)

    env.close()
    writer.close()

    return scores, avg_scores


def visualize_performance(episodes, scores, avg_scores, filename):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(episodes, scores, label="Score", alpha=0.6)
    ax.plot(episodes, avg_scores, label="Average Score", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("DDPG Performance")
    ax.legend()

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
