import logging
import logging.handlers
import json
from datetime import datetime, timezone
import os
from src.config import LOG_DIR


class CustomLogger:
    def __init__(self, name="customLogger", log_dir=LOG_DIR):
        # Configure a named logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)

        # Create the logs directory if it doesn't exist
        logs_dir = log_dir
        os.makedirs(logs_dir, exist_ok=True)

        # Create a file handler that logs even debug messages
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(logs_dir, "logs.json"), maxBytes=5 * 1024 * 1024, backupCount=5
        )

        # Create and set the custom JSON formatter
        formatter = JSONFormatter()
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        self._logger.addHandler(file_handler)

    @property
    def logger(self):
        # This is a getter for the logger
        return self._logger


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
