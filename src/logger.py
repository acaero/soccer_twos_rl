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
    def __init__(self, name="Please Set Name"):

        self.name = name

        self._logger = logging.getLogger()
        self._logger.setLevel(logging.INFO)

        self.checkpoint_dir = Path(CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.tensorboard_log_dir = Path(LOG_DIR) / "tensorboard"
        self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

        self.plain_log_dir = Path(LOG_DIR) / "plain"
        self.plain_log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.tensorboard_log_dir)
        self.scores = []

        # Create a file handler that logs even debug messages
        log_path = Path(self.plain_log_dir) / "logs.json"
        file_handler = logging.FileHandler(log_path)

        # Create and set the custom JSON formatter
        formatter = JSONFormatter()
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        self._logger.addHandler(file_handler)

    def write_logs_and_tensorboard(
        self, iteration, scores, obs, reward, done, info, agent, custom={"action": None}
    ):

        avg_score = np.mean(
            [score for i, score in enumerate(scores.values()) if i < agent.num_agents]
        )
        # Log metrics to TensorBoard
        # TODO: Add more metrics to TensorBoard
        for i in range(len(scores)):
            self.writer.add_scalar(f"Score of {i} agent", scores[i], iteration)
        self.writer.add_scalar(
            f"Average Score of first {agent.num_agents} agent/s", avg_score, iteration
        )

        self._logger.info(
            f"{self.name}",
            extra={
                "custom_fields": {
                    "episode": iteration,
                    "scores": scores,
                    "reward": str(convert_arrays_to_lists(reward)),
                    "done": str(convert_arrays_to_lists(done)),
                    "info": str(convert_arrays_to_lists(info)),
                    "observations": str(convert_arrays_to_lists(obs)),
                    **custom,
                }
            },
        )

        # Save the best model
        if not self.scores or max(self.scores) < avg_score:
            self.scores = [avg_score]
            print("new best found at iteration", iteration)
            # Search for existing best model file and delete it
            best_model_pattern = f"best_model_{self.name}_*.pth"
            existing_best_model_files = list(
                self.checkpoint_dir.glob(best_model_pattern)
            )
            if existing_best_model_files:
                for file in existing_best_model_files:
                    file.unlink()  # Delete existing best model file

            # Save the new best model
            best_model_path = (
                self.checkpoint_dir / f"best_model_{self.name}_{iteration}.pth"
            )
            agent.save(best_model_path)

        self.scores.append(avg_score)

        # Save model checkpoint every 10 episodes
        if iteration % 10 == 0:
            # Print the custom fields as well
            custom_str = ", ".join(f"{key}: {value}" for key, value in custom.items())
            print(
                f"Episode: {iteration}, Average Score of first {agent.num_agents} agents: {avg_score:.2f}, Name: {self.name}, {custom_str}"
            )
            checkpoint_filename = (
                self.checkpoint_dir / f"checkpoint_{self.name}_{iteration}.pth"
            )
            agent.save(checkpoint_filename)


class JSONFormatter(logging.Formatter):
    def format(self, record):
        # Use the log record's created time for the timestamp
        timestamp = datetime.fromtimestamp(record.created, timezone.utc).isoformat()

        log_record = {
            "timestamp": timestamp,
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Include exception info if present
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)

        # Include stack trace if present
        if record.stack_info:
            log_record["stack_info"] = self.formatStack(record.stack_info)

        # Add custom fields if they exist
        if hasattr(record, "custom_fields"):
            log_record.update(record.custom_fields)

        return json.dumps(log_record, default=str)
