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
