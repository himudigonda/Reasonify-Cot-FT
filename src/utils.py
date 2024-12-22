import logging
import os
from datetime import datetime

from rich.logging import RichHandler
from rich.console import Console
from rich import print as rprint


def setup_logging(log_dir, log_level=logging.INFO):
    """Sets up logging for the project."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    console = Console()
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[
            RichHandler(rich_tracebacks=True, console=console),
            logging.FileHandler(log_file),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger
