import logging
import logging.handlers
import json
from datetime import datetime, timezone

class CustomLogger():
    def __init__(self) -> None:
        # Configure the root logger
        self._logger = logging.getLogger()
        self._logger.setLevel(logging.DEBUG)

        # Create a file handler that logs even debug messages
        file_handler = logging.handlers.RotatingFileHandler(
            'src/logs/logs.json', maxBytes=5*1024*1024, backupCount=5
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def format(self, record):
        log_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),  # Corrected line
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'filename': record.pathname,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add custom fields if they exist
        if hasattr(record, 'custom_fields'):
            log_record.update(record.custom_fields)
        
        return json.dumps(log_record)


