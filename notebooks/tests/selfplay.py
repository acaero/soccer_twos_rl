from pathlib import Path
import gym
import numpy as np
import soccer_twos
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import random
import math
import os
from src.utils import shape_rewards, decompress_floats, compress_floats
from src.config import N_GAMES, LOG_DIR, CHECKPOINT_DIR
from src.logger import CustomLogger


class SoccerTwosEnv(gym.Env):
    def __init__(self, worker_id=0, render=False, logger=None, learning_agent=-1):
        self.env = soccer_twos.make(worker_id=worker_id, render=render)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.worker_id = worker_id
        self.logger = logger
        self.iteration = 0
        self.learning_agent = learning_agent

    def reset(self):
        out = self.env.reset()
        if self.learning_agent == 0 or self.learning_agent == 2:
            print(
                "obs",
                len(out),
                "obs[self.learning_agent]",
                len(out[self.learning_agent]),
            )
        return out if self.learning_agent == -1 else out[self.learning_agent]

    def step(self, action):
        actions = {0: action, 1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0]}

        if len(action) == 2:
            actions[0] = np.array(action[0])
            actions[2] = np.array(action[1])

        obs, rewards, dones, info = self.env.step(actions)
        self.iteration += 1

        if self.logger is not None:
            self.logger.write_logs_and_tensorboard(
                self.iteration,
                {i: shape_rewards(info, i) for i in range(4)},
                obs,
                rewards,
                dones["__all__"],
                info,
                actions,
            )
            if self.learning_agent == 0 or self.learning_agent == 2:
                print(
                    "obs",
                    len(obs),
                    "obs[self.learning_agent]",
                    len(obs[self.learning_agent]),
                )
        return (
            obs if self.learning_agent == -1 else obs[self.learning_agent],
            (
                float(rewards[0]) + shape_rewards(info, 0)
                if len(action) == 2
                else compress_floats(rewards[0], rewards[2])
            ),
            dones["__all__"],
            info,
        )

    def close(self):
        return self.env.close()


class SelfPlayEnv(gym.Env):
    def __init__(self, base_env, model1, model2):
        self.base_env = base_env
        self.model1 = model1
        self.model2 = model2
        self.current_player = 0
        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space
        self.reward_range = (-float("inf"), float("inf"))

    def reset(self):
        self.current_player = 0
        obs = self.base_env.reset()
        return obs[self.current_player]

    def step(self, action):
        if self.current_player == 0:
            action2, _ = self.model2.predict(
                self.base_env.env.observation[2], deterministic=True
            )
            actions = {0: action, 1: [0, 0, 0], 2: action2, 3: [0, 0, 0]}
        else:
            action1, _ = self.model1.predict(
                self.base_env.env.observation[0], deterministic=True
            )
            actions = {0: action1, 1: [0, 0, 0], 2: action, 3: [0, 0, 0]}

        obs, rewards, done, info = self.base_env.step(actions)

        reward = rewards[self.current_player]
        self.current_player = 1 - self.current_player

        return obs[self.current_player], reward, done, info

    def close(self):
        return self.base_env.close()


class ProgressBarCallback(BaseCallback):
    def __init__(self, pbar):
        super().__init__()
        self._pbar = pbar

    def _on_step(self):
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


class ProgressBarManager(object):
    def __init__(self, total_timesteps):
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):
        self.pbar = tqdm(total=self.total_timesteps)
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


class BestModelCallback(BaseCallback):
    def __init__(self, save_path, verbose=1):
        super(BestModelCallback, self).__init__(verbose)
        self.save_path = save_path
        self.best_elo = -float("inf")

    def _on_step(self):
        return True

    def on_rollout_end(self):
        current_elo = self.model.elo
        if current_elo > self.best_elo:
            self.best_elo = current_elo
            path = os.path.join(self.save_path, f"best_model_elo_{self.best_elo:.0f}")
            self.model.save(path)
            if self.verbose > 0:
                print(f"Saving new best model with Elo {self.best_elo:.0f}")


def calculate_expected_score(rating1, rating2):
    return 1 / (1 + math.pow(10, (rating2 - rating1) / 400))


def update_elo(rating1, rating2, score, k_factor=32):
    expected = calculate_expected_score(rating1, rating2)
    new_rating = rating1 + k_factor * (score - expected)
    return new_rating


def evaluate_agents(env, model1, model2, n_episodes=10):
    total_reward1 = 0
    total_reward2 = 0

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action1, _ = model1.predict(obs[0])
            action2, _ = model2.predict(obs[2])
            actions = {0: action1, 1: [0, 0, 0], 2: action2, 3: [0, 0, 0]}
            obs, rewards, done, _ = env.step(actions)
            total_reward1 += rewards[0]
            total_reward2 += rewards[2]
            done = done

    return total_reward1 / n_episodes, total_reward2 / n_episodes


def self_play(agent_pool, base_env, n_games, n_steps_per_game, save_path):
    callbacks = [BestModelCallback(save_path=save_path)]

    for _ in range(n_games):
        agent1, agent2 = random.sample(agent_pool, 2)

        model1, elo1 = agent1
        model2, elo2 = agent2

        # Create a self-play environment
        self_play_env = SelfPlayEnv(base_env, model1, model2)

        # Train both models in the self-play environment
        model1.set_env(self_play_env)
        model1.learn(
            total_timesteps=n_steps_per_game,
            reset_num_timesteps=False,
            callback=callbacks,
        )

        # Reset the environment and switch roles
        self_play_env = SelfPlayEnv(base_env, model2, model1)  # Switch roles
        model2.set_env(self_play_env)
        model2.learn(
            total_timesteps=n_steps_per_game,
            reset_num_timesteps=False,
            callback=callbacks,
        )

        # Evaluate the models
        total_reward1, total_reward2 = evaluate_agents(base_env, model1, model2)

        if total_reward1 > total_reward2:
            score = 1
        elif total_reward1 < total_reward2:
            score = 0
        else:
            score = 0.5

        new_elo1 = update_elo(elo1, elo2, score)
        new_elo2 = update_elo(elo2, elo1, 1 - score)

        print(
            f"Game over: {total_reward1:.2f} vs {total_reward2:.2f}, Elo: {new_elo1:.0f} vs {new_elo2:.0f}"
        )

        agent_pool.remove(agent1)
        agent_pool.remove(agent2)
        model1.elo = new_elo1
        model2.elo = new_elo2
        agent_pool.append((model1, new_elo1))
        agent_pool.append((model2, new_elo2))

    return agent_pool


if __name__ == "__main__":
    print("Initializing...")
    base_env = SoccerTwosEnv(worker_id=0, render=False)

    print("Creating agent pool...")
    pool_size = 5
    agent_pool = [
        (
            PPO(
                "MlpPolicy",
                base_env,
                verbose=1,
                tensorboard_log=LOG_DIR + "/tensorboard/",
                device="auto",
            ),
            1500,
        )
        for _ in range(pool_size)
    ]

    n_steps_per_game = 10_000
    n_games = 5

    save_path = Path(CHECKPOINT_DIR) / "self_play_models"
    save_path.mkdir(parents=True, exist_ok=True)

    print("Starting self-play training...")
    with ProgressBarManager(n_steps_per_game * n_games * 2) as progress_callback:
        updated_agent_pool = self_play(
            agent_pool, base_env, n_games, n_steps_per_game, save_path
        )

    print("Final Elo ratings:")
    for i, (model, elo) in enumerate(
        sorted(updated_agent_pool, key=lambda x: x[1], reverse=True)
    ):
        print(f"Agent {i+1}: Elo {elo:.0f}")
