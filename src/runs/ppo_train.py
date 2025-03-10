from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.env import make_env, ProgressBarManager
from src.config import N_GAMES, LOG_DIR, CHECKPOINT_DIR


RUN_NAME = "ppo_speed_final_single"

if __name__ == "__main__":
    num_cpu = 6
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
        gamma=0.999,
    )
    reset = False
    # model = PPO.load(
    #     path=r"out\checkpoints\bestmodel_ppo.zip", env=vec_env, device="auto", verbose=1
    # )
    # model.set_parameters(load_path_or_dict=r"out\checkpoints\bestmodel_ppo.zip")
    # reset = True

    with ProgressBarManager(N_GAMES) as progress_callback:
        model.learn(N_GAMES, callback=[progress_callback])
    model.save(Path(CHECKPOINT_DIR) / "ppo_speed_final_single")
