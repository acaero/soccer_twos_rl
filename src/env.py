import gym
import numpy as np
import soccer_twos

from tqdm import tqdm

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback

from src.utils import shape_rewards
from src.logger import CustomLogger
from src.agents.baseline_agent import BaselineAgent


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
        actions = {0: action, 1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0]}

        obs, rewards, dones, info = self.env.step(actions)

        # kommentiere die 5 Zeilen aus für ohne BaselineAgent
        # agent = BaselineAgent()
        # actions = {player_id: agent.act(info, player_id) for player_id in range(4)}
        # actions[0] = action
        # self.env.reset()
        # obs, rewards, dones, info = self.env.step(actions)

        self.iteration += 1
        if self.logger is not None:
            self.logger.write_logs_and_tensorboard(
                self.iteration,
                {i: shape_rewards(info, i) for i in range(4)},
                obs,
                rewards,
                dones["__all__"],
                info,
                action,
            )
        return (
            np.array(obs[0]),
            float(rewards[0]) + shape_rewards(info, 0),
            dones["__all__"],
            info,
        )

    def close(self):
        return self.env.close()


def make_env(env_id: int, rank: int, seed: int = 0, tensorboard_name: str = "default"):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """

    def _init():
        if rank == 0:
            logger = CustomLogger(
                "ppo", run_name=f"{tensorboard_name}_{rank}", save=False
            )
        else:
            logger = None
        env = SoccerTwosEnv(worker_id=env_id, render=False, logger=logger)
        env.reset()
        return env

    set_random_seed(seed)
    return _init


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super().__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()
