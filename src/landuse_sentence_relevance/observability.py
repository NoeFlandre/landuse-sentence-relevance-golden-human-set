from __future__ import annotations

import logging
from logging import Logger
from typing import Final

STREAM_PROGRESS_INTERVAL: Final = 10_000
DEPENDENCY_LOGGERS: Final = ("httpx", "datasets", "huggingface_hub")


def configure_logging() -> None:
    """Configure concise INFO logs for the command-line annotation process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for logger_name in DEPENDENCY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def log_stream_progress(logger: Logger, label: str, rows_seen: int) -> None:
    """Report the first streamed row and sparse checkpoints without logging row data."""
    if rows_seen == 1 or rows_seen % STREAM_PROGRESS_INTERVAL == 0:
        noun = "row" if rows_seen == 1 else "rows"
        logger.info("%s: scanned %s %s", label, f"{rows_seen:,}", noun)
