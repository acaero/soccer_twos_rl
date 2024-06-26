from src.utils import convert_arrays_to_lists, RewardShaper
from src.config import CHECKPOINT_DIR, N_GAMES, IMAGES_DIR, LOG_DIR
import numpy as np
import soccer_twos
from tqdm import tqdm
from pathlib import Path
from src.logger import CustomLogger
from src.agents.maddpg_agent import MADDPGAgent
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def train_maddpg(
    n_games=N_GAMES,
    checkpoint_dir=CHECKPOINT_DIR,
    log_dir=LOG_DIR,
    images_dir=IMAGES_DIR,
    mode="single",
):
    env = soccer_twos.make(render=False)
    reward_shaper = RewardShaper()

    if mode == "single":
        num_agents = 1
        agent_indices = [0]
    elif mode == "team":
        num_agents = 2
        agent_indices = [0, 1]
    elif mode == "all":
        num_agents = 4
        agent_indices = [0, 1, 2, 3]
    else:
        raise ValueError("Invalid mode. Choose from 'single', 'team', or 'all'.")

    state_size = 336
    action_size = 3
    maddpg_agent = MADDPGAgent(num_agents, state_size, action_size)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_log_dir = Path(log_dir) / "tensorboard"
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    plain_log_dir = Path(log_dir) / "plain"
    plain_log_dir.mkdir(parents=True, exist_ok=True)
    logger = CustomLogger(log_dir=plain_log_dir).logger

    team1_scores, team2_scores = [], []
    avg_team1_scores, avg_team2_scores = [], []

    # Collect initial experiences
    initial_experiences = 1000
    experiences_count = 0
    obs = env.reset()

    while experiences_count < initial_experiences:
        actions = maddpg_agent.act({i: obs[i] for i in agent_indices})
        for i in range(len(agent_indices), 4):
            actions[i] = [0, 0, 0]

        next_obs, rewards, dones, info = env.step(actions)
        for agent_id in agent_indices:
            shaped_reward = reward_shaper.calculate_reward(
                obs[agent_id], next_obs[agent_id], info, int(agent_id)
            )
            rewards[agent_id] *= 10
            rewards[agent_id] += shaped_reward
        done = dones["__all__"]
        maddpg_agent.remember(
            {i: obs[i] for i in agent_indices},
            {i: actions[i] for i in agent_indices},
            {i: rewards[i] for i in agent_indices},
            {i: next_obs[i] for i in agent_indices},
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
        team1_score = 0
        team2_score = 0
        episode_loss = {agent_id: 0 for agent_id in agent_indices}
        while not done:
            actions = maddpg_agent.act({i: obs[i] for i in agent_indices})
            for i in range(len(agent_indices), 4):
                actions[i] = [0, 0, 0]

            next_obs, rewards, dones, info = env.step(actions)
            done = dones["__all__"]

            for agent_id in agent_indices:
                shaped_reward = reward_shaper.calculate_reward(
                    obs[agent_id], next_obs[agent_id], info, int(agent_id)
                )
                rewards[agent_id] += shaped_reward

            maddpg_agent.remember(
                {i: obs[i] for i in agent_indices},
                {i: actions[i] for i in agent_indices},
                {i: rewards[i] for i in agent_indices},
                {i: next_obs[i] for i in agent_indices},
                dones,
            )
            loss = maddpg_agent.replay()
            if loss is not None:
                for agent_id in agent_indices:
                    episode_loss[agent_id] += loss[agent_id]

            obs = next_obs
            team1_score += rewards[0] + rewards[1]
            team2_score += rewards[2] + rewards[3]

        team1_scores.append(team1_score)
        team2_scores.append(team2_score)
        avg_team1_score = np.mean(team1_scores[-100:])
        avg_team2_score = np.mean(team2_scores[-100:])
        avg_team1_scores.append(avg_team1_score)
        avg_team2_scores.append(avg_team2_score)

        # Log metrics to TensorBoard
        writer.add_scalar("Team1/Score", team1_score, i)
        writer.add_scalar("Team2/Score", team2_score, i)
        writer.add_scalar("Team1/Average Score", avg_team1_score, i)
        writer.add_scalar("Team2/Average Score", avg_team2_score, i)
        for agent_id in agent_indices:
            writer.add_scalar(
                f"Loss/Agent{agent_id} Episode Loss", episode_loss[agent_id], i
            )

        # Log metrics to plain log file
        logger.info(
            "MADDPG Agent",
            extra={
                "custom_fields": {
                    "episode": i,
                    "team1_score": team1_score,
                    "team2_score": team2_score,
                    "average_team1_score": avg_team1_score,
                    "average_team2_score": avg_team2_score,
                    "reward": str(convert_arrays_to_lists(rewards)),
                    "done": str(convert_arrays_to_lists(dones)),
                    "info": str(convert_arrays_to_lists(info)),
                    "observations": str(convert_arrays_to_lists(obs)),
                }
            },
        )

        if i % 10 == 0:
            print(
                f"Episode: {i}, Team 1 Score: {team1_score:.2f}, Team 2 Score: {team2_score:.2f}, Avg Team 1 Score: {avg_team1_score:.2f}, Avg Team 2 Score: {avg_team2_score:.2f}"
            )
            checkpoint_filename = checkpoint_dir / f"checkpoint_MADDPG_{i}.pth"
            maddpg_agent.save(checkpoint_filename)

    env.close()
    writer.close()

    # Save performance visualization
    visualize_performance(
        range(n_games),
        team1_scores,
        avg_team1_scores,
        team2_scores,
        avg_team2_scores,
        images_dir / f"average_reward_{mode}.png",
    )

    return team1_scores, team2_scores, avg_team1_scores, avg_team2_scores


def visualize_performance(
    episodes, team1_scores, avg_team1_scores, team2_scores, avg_team2_scores, filename
):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(episodes, team1_scores, label="Team 1 Score", alpha=0.6)
    ax.plot(episodes, avg_team1_scores, label="Average Team 1 Score", linewidth=2)
    ax.plot(episodes, team2_scores, label="Team 2 Score", alpha=0.6)
    ax.plot(episodes, avg_team2_scores, label="Average Team 2 Score", linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("MADDPG Performance")
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    mode = "single"  # Change this to 'team' or 'all' as needed
    team1_scores, team2_scores, avg_team1_scores, avg_team2_scores = train_maddpg(
        n_games=N_GAMES, mode=mode
    )
    episodes = list(range(1, len(team1_scores) + 1))
    image_dir = Path(IMAGES_DIR)
    image_dir.mkdir(parents=True, exist_ok=True)
    visualize_performance(
        episodes,
        team1_scores,
        avg_team1_scores,
        team2_scores,
        avg_team2_scores,
        image_dir / f"soccer_twos_maddpg_performance_{mode}.png",
    )
