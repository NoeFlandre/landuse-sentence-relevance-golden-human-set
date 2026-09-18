import logging

from landuse_sentence_relevance.observability import configure_logging, log_stream_progress


def test_configure_logging_uses_a_concise_timestamped_info_format(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: calls.append(kwargs))

    configure_logging()

    assert calls == [
        {
            "level": logging.INFO,
            "format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
            "datefmt": "%H:%M:%S",
        }
    ]


def test_configure_logging_quiets_dependency_request_logs() -> None:
    for logger_name in ("httpx", "datasets", "huggingface_hub"):
        logging.getLogger(logger_name).setLevel(logging.INFO)

    configure_logging()

    assert all(
        logging.getLogger(logger_name).level == logging.WARNING
        for logger_name in ("httpx", "datasets", "huggingface_hub")
    )


def test_stream_progress_logs_the_first_row_and_periodic_checkpoints(caplog) -> None:
    logger = logging.getLogger("test.stream")
    caplog.set_level(logging.INFO, logger=logger.name)

    log_stream_progress(logger, "Wikipedia polygons", 1)
    log_stream_progress(logger, "Wikipedia polygons", 2)
    log_stream_progress(logger, "Wikipedia polygons", 10_000)

    assert [record.message for record in caplog.records] == [
        "Wikipedia polygons: scanned 1 row",
        "Wikipedia polygons: scanned 10,000 rows",
    ]


def test_shard_progress_names_the_source_the_shard_and_the_rows(caplog) -> None:
    """An operator watching a five-hour stream needs to tell "slow" from "stuck" from "looping".

    Candidate counts cannot do it: re-reading shards a source already covered lands candidates in
    occupied strata, so the total sits still while real work happens. During one website leg the
    count held at exactly 3,052 for over five hours at 73-95% CPU.
    """
    from landuse_sentence_relevance.observability import log_shard_progress

    logger = logging.getLogger("test.shards")
    with caplog.at_level(logging.INFO, logger="test.shards"):
        log_shard_progress(logger, "website", shard_index=11, total_shards=386, rows_seen=4_000)

    (record,) = caplog.records
    assert "website" in record.getMessage()
    assert "12" in record.getMessage(), "shards are reported from 1, the way a reader counts them"
    assert "386" in record.getMessage()
    assert "4,000" in record.getMessage()


def test_shard_progress_reports_an_unknown_total_without_inventing_one(caplog) -> None:
    """The lazy path cannot count shards without consuming them, and consuming them to make a
    log line nicer would defeat the laziness the join index depends on."""
    from landuse_sentence_relevance.observability import log_shard_progress

    logger = logging.getLogger("test.shards")
    with caplog.at_level(logging.INFO, logger="test.shards"):
        log_shard_progress(logger, "wikipedia", shard_index=0, total_shards=None, rows_seen=12)

    (record,) = caplog.records
    assert "?" in record.getMessage()
    assert "wikipedia" in record.getMessage()


def test_shard_progress_logs_no_row_content(caplog) -> None:
    """The builder's output stays compact: no candidate sentences, no upstream rows."""
    from landuse_sentence_relevance.observability import log_shard_progress

    logger = logging.getLogger("test.shards")
    with caplog.at_level(logging.INFO, logger="test.shards"):
        log_shard_progress(logger, "description", shard_index=2, total_shards=9, rows_seen=1)

    message = caplog.records[0].getMessage()
    assert len(message) < 120, message
