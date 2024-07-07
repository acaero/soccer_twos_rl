import gym
import numpy as np
import soccer_twos

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env


class SoccerTwosEnv(gym.Env):
    def __init__(self, worker_id=0, render=False):
        self.env = soccer_twos.make(worker_id=worker_id, render=render)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        out = self.env.reset()
        return np.array(out[0])

    def step(self, action):
        action = {0: action, 1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0]}
        obs, rewards, dones, info = self.env.step(action)
        return np.array(obs[0]), float(rewards[0]), dones["__all__"], info

    # def render(self, mode='human'):
    #     return self.env.render(mode)

    def close(self):
        return self.env.close()


def make_env(env_id: int, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """

    def _init():
        env = SoccerTwosEnv(worker_id=env_id, render=False)
        env.reset()
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":

    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    vec_env = SubprocVecEnv([make_env(env_id=i + 1, rank=i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=25_000)

    obs = vec_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render()
