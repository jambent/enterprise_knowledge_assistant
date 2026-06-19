import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone


def setup_logging():
    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            ts = getattr(record, "timestamp_override", record.created)
            log_record = {
                "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() + "Z",
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }

            # Optional: include exception info
            if record.exc_info:
                log_record["exception"] = self.formatException(record.exc_info)

            return json.dumps(log_record)


    # -----------------------------
    # Configure RotatingFileHandler
    # -----------------------------
    handler = RotatingFileHandler(
        "app.log",      # file name
        maxBytes=1024,  # rotate after 1KB (demo)
        backupCount=3   # keep 3 backups
    )

    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
