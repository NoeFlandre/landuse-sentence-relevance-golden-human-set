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


def log_shard_progress(
    logger: Logger,
    label: str,
    *,
    shard_index: int,
    total_shards: int | None,
    rows_seen: int,
) -> None:
    """Report one completed shard: which source, which shard, how many rows it held.

    A source that runs for hours otherwise emits one line when it opens and one when its join
    index is built, and nothing in between. Candidate counts cannot fill the gap: re-reading
    shards a source already covered lands candidates in occupied strata, so the total sits still
    while real work happens -- during one website leg it held at exactly 3,052 for over five
    hours at 73-95% CPU, which is indistinguishable from stuck.

    Shards are numbered from 1 because that is how a reader counts them, and the total is
    ``?`` when it is not known: the sequential path reads shards lazily, and consuming them to
    make a log line nicer would defeat the laziness the join index depends on.

    One line per shard, so a 386-shard source emits 386 lines over the hours it runs. No row
    content, ever: the builder's output stays compact.
    """
    total = f"{total_shards}" if total_shards else "?"
    logger.info(
        "%s: finished shard %d/%s (%s rows)",
        label,
        shard_index + 1,
        total,
        f"{rows_seen:,}",
    )
