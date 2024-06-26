from src.utils import convert_arrays_to_lists, RewardShaper
from src.config import CHECKPOINT_DIR, N_GAMES, IMAGES_DIR, LOG_DIR
import numpy as np
import soccer_twos
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.agents.ddqn_agent import DDQNAgent
from src.logger import CustomLogger
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


def train_ddqn(
    n_games=N_GAMES, checkpoint_dir=CHECKPOINT_DIR, log_dir=LOG_DIR, mode="single"
):
    env = soccer_twos.make()
    reward_shaper = RewardShaper()

    scores, eps_history, avg_scores = [], [], []

    if mode == "single":
        ddqn_agents = [DDQNAgent(336, 3)]
        agent_indices = [0]
    elif mode == "team":
        ddqn_agents = [DDQNAgent(336, 3), DDQNAgent(336, 3)]
        agent_indices = [0, 1]
    elif mode == "all":
        ddqn_agents = [DDQNAgent(336, 3) for _ in range(4)]
        agent_indices = list(range(4))
    else:
        raise ValueError("Invalid mode. Choose from 'single', 'team', or 'all'.")

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
                if player_id in agent_indices:
                    actions[player_id] = ddqn_agents[
                        agent_indices.index(player_id)
                    ].act(obs[player_id])
                else:
                    actions[player_id] = [0, 0, 0]  # Static action for other agents

            next_obs, reward, done, info = env.step(actions)
            done = done["__all__"]

            for player_id in agent_indices:
                shaped_reward = reward[player_id] * 10 + reward_shaper.calculate_reward(
                    obs[player_id], next_obs[player_id], info, player_id
                )

                ddqn_agents[agent_indices.index(player_id)].remember(
                    obs[player_id],
                    actions[player_id],
                    shaped_reward,
                    next_obs[player_id],
                    done,
                )
                loss = ddqn_agents[agent_indices.index(player_id)].replay()
                if loss is not None:
                    episode_loss += loss

                score += shaped_reward

            obs = next_obs

        scores.append(score)
        eps_history.append(
            ddqn_agents[0].epsilon
        )  # Assume all agents share the same epsilon
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        # Log metrics to TensorBoard
        writer.add_scalar("Score", score, i)
        writer.add_scalar("Average Score", avg_score, i)
        writer.add_scalar("Epsilon", ddqn_agents[0].epsilon, i)
        writer.add_scalar("Loss/Episode Loss", episode_loss, i)

        # Log metrics to plain log file
        logger.info(
            "DDQN Agent",
            extra={
                "custom_fields": {
                    "episode": i,
                    "score": score,
                    "average_score": avg_score,
                    "epsilon": ddqn_agents[0].epsilon,
                    "reward": str(convert_arrays_to_lists(reward)),
                    "done": str(convert_arrays_to_lists(done)),
                    "info": str(convert_arrays_to_lists(info)),
                }
            },
        )

        if i % 10 == 0:
            print(
                f"Episode: {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, Epsilon: {ddqn_agents[0].epsilon:.2f}"
            )
            checkpoint_filename = checkpoint_dir / f"checkpoint_DDQN_{i}.pth"
            for agent_id, agent in enumerate(ddqn_agents):
                agent.save(
                    checkpoint_filename.with_name(
                        f"checkpoint_DDQN_agent_{agent_id}_{i}.pth"
                    )
                )

    env.close()
    writer.close()

    return scores, avg_scores, eps_history


def visualize_performance(episodes, scores, avg_scores, epsilons, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    ax1.plot(episodes, scores, label="Score", alpha=0.6)
    ax1.plot(episodes, avg_scores, label="Average Score", linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Score")
    ax1.set_title("DDQN Performance")
    ax1.legend()

    ax2.plot(episodes, epsilons)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Epsilon")
    ax2.set_title("Exploration Rate")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    mode = "team"  # Change this to 'single', 'team' or 'all' as needed
    scores, avg_scores, eps_history = train_ddqn(n_games=N_GAMES, mode=mode)
    episodes = list(range(1, len(scores) + 1))
    image_dir = Path(IMAGES_DIR)
    image_dir.mkdir(parents=True, exist_ok=True)
    visualize_performance(
        episodes,
        scores,
        avg_scores,
        eps_history,
        image_dir / f"soccer_twos_ddqn_performance_{mode}.png",
    )
