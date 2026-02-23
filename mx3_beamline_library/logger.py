import logging
import os


def setup_logger(name: str = None) -> logging.Logger:
    """
    Set up a logger for the mx3_beamline_library.

    Parameters
    ----------
    name : str
        Logger name. If None, uses the calling module's __name__

    Returns
    -------
        The logger instance
    """
    if name is None:
        logger_name = "mx3_beamline_library"
    else:
        logger_name = name
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger

    log_level = getattr(logging, os.getenv("BL_LOG_LEVEL", "INFO").upper())
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger
