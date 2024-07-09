from pathlib import Path
import gym
import numpy as np
import soccer_twos
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import random
import math
import os
from src.utils import shape_rewards
from src.config import N_GAMES, LOG_DIR, CHECKPOINT_DIR
from src.logger import CustomLogger


class SoccerTwosEnv(gym.Env):
    def __init__(self, worker_id=0, render=False, logger=None):
        self.env = soccer_twos.make(worker_id=worker_id, render=render)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.worker_id = worker_id
        self.logger = logger
        self.iteration = 0

    def reset(self):
        out = self.env.reset()
        return np.array(out[0])

    def step(self, action):
        action = {
            0: action,
            1: [[0, 0, 0]] * self.num_envs,
            2: [[0, 0, 0]] * self.num_envs,
            3: [[0, 0, 0]] * self.num_envs,
        }
        obs, rewards, dones, info = self.env.step(action)
        self.iteration += 1
        if self.logger is not None:
            self.logger.write_logs_and_tensorboard(
                self.iteration,
                {
                    i: [shape_rewards(info[j], i) for j in range(self.num_envs)]
                    for i in range(4)
                },
                obs,
                rewards,
                dones["__all__"],
                info,
                action,
            )
        return (
            np.array([obs[0] for obs in obs]),
            np.array(
                [
                    float(rewards[0][i]) + shape_rewards(info[i], 0)
                    for i in range(self.num_envs)
                ]
            ),
            np.array([dones["__all__"][i] for i in range(self.num_envs)]),
            info,
        )

    def close(self):
        return self.env.close()


def make_env(env_id: int, rank: int, seed: int = 0):
    def _init():
        if rank == 0:
            logger = CustomLogger("ppo", run_name=f"ppo_env{rank}", save=False)
        else:
            logger = None
        env = SoccerTwosEnv(worker_id=env_id, render=False, logger=logger)
        env.reset()
        return env

    set_random_seed(seed)
    return _init


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


def self_play(agent_pool, vec_env, n_games, n_steps_per_game, save_path):
    callbacks = [BestModelCallback(save_path=save_path)]

    for _ in range(n_games):
        agent1, agent2 = random.sample(agent_pool, 2)
        model1, elo1 = agent1
        model2, elo2 = agent2

        obs = vec_env.reset()

        total_reward1 = 0
        total_reward2 = 0

        for step in range(n_steps_per_game):
            if step % 2 == 0:
                action, _ = model1.predict(obs)
            else:
                action, _ = model2.predict(obs)

            obs, reward, done, info = vec_env.step(action)

            if step % 2 == 0:
                total_reward1 += reward[
                    0
                ]  # Assuming we're interested in the first environment
            else:
                total_reward2 += reward[0]

            if done.any():  # Check if any environment is done
                break

        if total_reward1 > total_reward2:
            score = 1
        elif total_reward1 < total_reward2:
            score = 0
        else:
            score = 0.5

        new_elo1 = update_elo(elo1, elo2, score)
        new_elo2 = update_elo(elo2, elo1, 1 - score)

        agent_pool.remove(agent1)
        agent_pool.remove(agent2)
        model1.elo = new_elo1
        model2.elo = new_elo2
        agent_pool.append((model1, new_elo1))
        agent_pool.append((model2, new_elo2))

        model1.learn(
            total_timesteps=n_steps_per_game,
            reset_num_timesteps=False,
            callback=callbacks,
        )
        model2.learn(
            total_timesteps=n_steps_per_game,
            reset_num_timesteps=False,
            callback=callbacks,
        )

    return agent_pool


if __name__ == "__main__":
    num_cpu = 4
    pool_size = 10
    vec_env = SubprocVecEnv([make_env(env_id=i + 1, rank=i) for i in range(num_cpu)])

    agent_pool = [
        (
            PPO(
                "MlpPolicy",
                vec_env,
                verbose=0,
                tensorboard_log=LOG_DIR + "/tensorboard/",
                device="auto",
            ),
            1500,
        )
        for _ in range(pool_size)
    ]

    n_steps_per_game = 1000

    save_path = Path(CHECKPOINT_DIR) / "self_play_models"
    save_path.mkdir(parents=True, exist_ok=True)

    with ProgressBarManager(N_GAMES) as progress_callback:
        updated_agent_pool = self_play(
            agent_pool, vec_env, N_GAMES, n_steps_per_game, save_path
        )

    for i, (model, elo) in enumerate(
        sorted(updated_agent_pool, key=lambda x: x[1], reverse=True)
    ):
        print(f"Agent {i+1}: Elo {elo:.0f}")
