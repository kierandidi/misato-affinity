import logging
import logging.handlers
import os
import sys

from pytorch_lightning.loggers import WandbLogger

from src.constants import DEFAULT_LOG_FILE, DEFAULT_LOG_FORMATTER, DEFAULT_LOG_LEVEL


def _get_console_handler(
    stream=sys.stdout,
    formatter: logging.Formatter = DEFAULT_LOG_FORMATTER,
) -> logging.StreamHandler:
    """Returns Handler that prints to stdout."""
    console_handler = logging.StreamHandler(stream)
    console_handler.setFormatter(formatter)
    return console_handler


def _get_timed_file_handler(
    log_file: os.PathLike = DEFAULT_LOG_FILE,
    formatter: logging.Formatter = DEFAULT_LOG_FORMATTER,
    when: str = "midnight",
    backup_count: int = 7,
    **timed_rotating_file_handler_kwargs,
) -> logging.handlers.TimedRotatingFileHandler:
    """Returns Handler that keeps logs for specified amount of time."""
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file, when=when, backupCount=backup_count, **timed_rotating_file_handler_kwargs
    )
    file_handler.setFormatter(formatter)
    return file_handler


def get_logger(
    logger_name: str,
    level: int = DEFAULT_LOG_LEVEL,
    propagate: bool = False,
    log_to_console: bool = True,
    log_to_file: bool = True,
    **handler_kwargs,
) -> logging.Logger:
    """Returns logger with console and timed file handler."""

    logger = logging.getLogger(logger_name)

    # if logger already has handlers attached to it, skip the configuration
    if logger.hasHandlers():
        logger.debug("Logger %s already set up.", logger.name)
        return logger

    logger.setLevel(level)

    if log_to_console:
        logger.addHandler(_get_console_handler(**handler_kwargs))
    if log_to_file:
        logger.addHandler(_get_timed_file_handler(**handler_kwargs))

    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = propagate

    return logger


def get_lightning_logger(cfg):
    if cfg.name == "test":  # Returns no logger if a test/debugging
        return None

    # Setup logging
    if cfg.logger.name == "wandb":
        logger = WandbLogger(
            name=cfg.name,
            save_dir=cfg.hydra.run.dir,
            entity="enter_entity_name",
            project="enter_project_name",
            save_code=False,
        )
    else:
        raise NotImplementedError("Logger not implemented yet")

    return logger
