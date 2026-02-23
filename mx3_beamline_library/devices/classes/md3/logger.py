"""
MD3 Logger interface using mx3_beamline_library logger.
Replaces the original colored logging implementation.
"""

from mx3_beamline_library.logger import setup_logger

# Get the logger for MD3 devices
logger = setup_logger("mx3_beamline_library.md3")

# Status constants for backward compatibility
OK = 1
FAILED = -1
NA = 0
DEBUG = 2


def trace(msg):
    """Trace function - logs at debug level"""
    logger.debug(msg)


def log(msg, success=NA):
    """
    Log the result of a test with appropriate log level based on success status

    Args:
        msg: Message to log
        success: Status indicator (OK, FAILED, NA, DEBUG, or string "DEBUG")
    """
    if success is NA:
        logger.info(msg)
    elif success is OK:
        logger.info(f"SUCCESS: {msg}")
    elif success is FAILED:
        logger.error(f"FAILED: {msg}")
    elif success is DEBUG or success == "DEBUG":
        logger.debug(msg)
    else:
        # Handle any other case
        logger.info(msg)
