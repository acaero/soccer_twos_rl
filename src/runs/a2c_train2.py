from pathlib import Path

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.env import make_env, ProgressBarManager
from src.config import N_GAMES, LOG_DIR, CHECKPOINT_DIR


RUN_NAME = "a2c_v5"

if __name__ == "__main__":
    num_cpu = 4
    vec_env = SubprocVecEnv(
        [
            make_env(env_id=i + 1, rank=i, tensorboard_name=RUN_NAME)
            for i in range(num_cpu)
        ]
    )

    # model = A2C.load(r"C:\Users\echte\OneDrive\Dokumente\DHBW\Semester 6\RL 2\soccer_twos_rl\src\runs\out\checkpoints\bestmodel_a2c.zip")
    model = A2C(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR + "/tensorboard/",
        device="auto",
    )

    with ProgressBarManager(100_000) as progress_callback:
        model.learn(100_000, callback=[progress_callback])
    model.save(Path(CHECKPOINT_DIR) / "bestmodel_a2c_5")
