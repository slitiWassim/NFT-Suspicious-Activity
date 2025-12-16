import logging
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------
# DEFINE LOGGER FUNCTION 
# -------------------------------------------------------------------
import logging
from pathlib import Path

def setup_logger(log_path):
    logger = logging.getLogger("temporal_cycles")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger  # IMPORTANT: avoid duplicate handlers

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter(
        "%(asctime)s | %(processName)s | %(levelname)s | %(message)s"
    )
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.propagate = False

    return logger

