from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.env import make_env, ProgressBarManager
from src.config import N_GAMES, LOG_DIR, CHECKPOINT_DIR


RUN_NAME = "ppo_v1"

if __name__ == "__main__":
    num_cpu = 4
    vec_env = SubprocVecEnv(
        [
            make_env(env_id=i + 1, rank=i, tensorboard_name=RUN_NAME)
            for i in range(num_cpu)
        ]
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR + "/tensorboard/",
        device="auto",
    )

    with ProgressBarManager(N_GAMES) as progress_callback:
        model.learn(N_GAMES, callback=[progress_callback])
    model.save(Path(CHECKPOINT_DIR) / "bestmodel_ppo2")
