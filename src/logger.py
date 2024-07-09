import json
from datetime import datetime, timezone
from src.config import CHECKPOINT_DIR, LOG_DIR
from src.utils import convert_arrays_to_lists
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np


class CustomLogger:
    def __init__(self, name="Please Set Name", run_name: str = "default", save=True):
        self.name = name
        self.run_name = run_name
        self.save = save

        self.checkpoint_dir = Path(CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard logs
        self.tensorboard_log_dir = Path(LOG_DIR) / "tensorboard" / run_name
        self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

        # Plain logs
        self.plain_log_dir = Path(LOG_DIR) / "plain" / run_name
        self.plain_log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.tensorboard_log_dir)
        self.scores = []

        self.log_file = self.plain_log_dir / f"logs_{self.name}.json"

    def log(self, **kwargs):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "name": self.name,
            **kwargs,
        }
        with open(self.log_file, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")  # Add a newline for readability

    def write_logs_and_tensorboard(
        self,
        iteration,
        scores,
        obs,
        rewards,
        done,
        info,
        actions,
        agent=None,
        custom={"": None},
    ):
        # Check if we're dealing with multi-environment or single-environment case
        multi_env = isinstance(scores[0], (np.ndarray, list))

        if multi_env:
            avg_scores = {i: np.mean(score) for i, score in scores.items()}
            avg_score = np.mean([score for i, score in avg_scores.items()])
        else:
            avg_scores = scores
            avg_score = np.mean([score for _, score in scores.items()])

        # Log metrics to TensorBoard
        self.writer.add_scalar(
            f"Difference of agent 0 score to average score",
            scores[0] - avg_score,
            iteration,
        )
        for i in range(len(scores)):
            self.writer.add_scalar(f"Score of agent {i}", scores[i], iteration)
            self.writer.add_scalar(f"Reward of agent {i}", rewards[i], iteration)

        # Log to file
        self.log(
            episode=iteration,
            scores=str(convert_arrays_to_lists(scores)),
            reward=str(convert_arrays_to_lists(rewards)),
            done=str(convert_arrays_to_lists(done)),
            info=str(convert_arrays_to_lists(info)),
            actions=str(convert_arrays_to_lists(actions)),
            observations=str(convert_arrays_to_lists(obs)),
            **custom,
        )

        # Save the best model
        if not self.scores or max(self.scores) < avg_score:
            self.scores = [avg_score]

            best_model_pattern = f"best_model_{self.name}_*.pth"
            existing_best_model_files = list(
                self.checkpoint_dir.glob(best_model_pattern)
            )
            if existing_best_model_files:
                for file in existing_best_model_files:
                    file.unlink()  # Delete existing best model file

            best_model_path = (
                self.checkpoint_dir
                / f"best_model_{self.name}_{iteration}_{self.run_name}.pth"
            )
            if self.save and agent:
                agent.save(best_model_path)

        self.scores.append(avg_score)

        # Save model checkpoint every 10 episodes
        if iteration % 10 == 0:
            checkpoint_filename = (
                self.checkpoint_dir
                / f"checkpoint_{self.name}_{iteration}_{self.run_name}.pth"
            )
            if self.save and agent:
                agent.save(checkpoint_filename)

    def log_and_print(self, message, **kwargs):
        self.log(message, **kwargs)
        print(f"{self.name}: {message}")
