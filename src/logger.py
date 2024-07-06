import logging
import logging.handlers
import json
from datetime import datetime, timezone
from src.config import CHECKPOINT_DIR, LOG_DIR
from src.utils import convert_arrays_to_lists
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np


class CustomLogger:
    def __init__(self, name="Please Set Name", run_name: str = "default"):
        self.name = name
        self.run_name = run_name

        self._logger = logging.getLogger()
        self._logger.setLevel(logging.INFO)

        self.checkpoint_dir = Path(CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.tensorboard_log_dir = Path(LOG_DIR) / "tensorboard" / run_name
        self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

        self.plain_log_dir = Path(LOG_DIR) / "plain"
        self.plain_log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.tensorboard_log_dir)
        self.scores = []

        log_path = Path(self.plain_log_dir) / "logs.json"
        file_handler = logging.FileHandler(log_path)

        formatter = JSONFormatter()
        file_handler.setFormatter(formatter)

        self._logger.addHandler(file_handler)

    def write_logs_and_tensorboard(
        self,
        iteration,
        scores,
        obs,
        reward,
        done,
        info,
        actions,
        agent,
        custom={"": None},
    ):
        # Check if we're dealing with multi-environment or single-environment case
        multi_env = isinstance(scores[0], (np.ndarray, list))

        if multi_env:
            avg_scores = {i: np.mean(score) for i, score in scores.items()}
            avg_score = np.mean(
                [score for i, score in avg_scores.items() if i < agent.num_agents]
            )
        else:
            avg_scores = scores
            avg_score = np.mean(
                [score for i, score in scores.items() if i < agent.num_agents]
            )

        # Log metrics to TensorBoard
        for i in range(len(avg_scores)):
            self.writer.add_scalar(f"Score of agent {i}", avg_scores[i], iteration)
            self.writer.add_scalar(f"Reward of agent {i}", avg_scores[i], iteration)

        self.writer.add_scalar(
            f"Average Score of first {agent.num_agents} agent/s", avg_score, iteration
        )

        self._logger.info(
            f"{self.name}",
            extra={
                "custom_fields": {
                    "episode": iteration,
                    "scores": str(convert_arrays_to_lists(scores)),
                    "reward": str(convert_arrays_to_lists(reward)),
                    "done": str(convert_arrays_to_lists(done)),
                    "info": str(convert_arrays_to_lists(info)),
                    "actions": str(convert_arrays_to_lists(actions)),
                    "observations": str(convert_arrays_to_lists(obs)),
                    **custom,
                }
            },
        )

        # Save the best model
        if not self.scores or max(self.scores) < avg_score:
            self.scores = [avg_score]
            print("new best found at iteration", iteration)
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
            agent.save(best_model_path)

        self.scores.append(avg_score)

        # Save model checkpoint every 500 episodes
        if iteration % 500 == 0:
            custom_str = ", ".join(f"{key}: {value}" for key, value in custom.items())
            print(
                f"Episode: {iteration}, Average Score of first {agent.num_agents} agents: {avg_score:.2f}, Name: {self.name}, {custom_str}"
            )
            checkpoint_filename = (
                self.checkpoint_dir
                / f"checkpoint_{self.name}_{iteration}_{self.run_name}.pth"
            )
            agent.save(checkpoint_filename)


class JSONFormatter(logging.Formatter):
    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created, timezone.utc).isoformat()

        log_record = {
            "timestamp": timestamp,
            "level": record.levelname,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info:
            log_record["stack_info"] = self.formatStack(record.stack_info)

        if hasattr(record, "custom_fields"):
            log_record.update(record.custom_fields)

        return json.dumps(log_record, default=str)
