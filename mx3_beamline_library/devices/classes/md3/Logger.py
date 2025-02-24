"""
This code is provided AS IS for example purpose and testing MD Device Server
ARINAX Sep. 2021
"""

import logging
import sys

import coloredlogs

coloredlogs.DEFAULT_LOG_FORMAT = "%(asctime)-19s %(message)-99s "
coloredlogs.COLOREDLOGS_LOG_FORMAT = "%(asctime)-19s %(message)-99s : %(state)10s"
coloredlogs.install(milliseconds=True, level="INFO")
FILE_FORMAT = "%(asctime)-19s : %(message)-90s : %(state)s"

# Every log
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(coloredlogs.ColoredFormatter())
logger = logging.getLogger("Info")
# rfh = RotatingFileHandler('{:%Y-%m-%d %H%M%S}.log'.format(datetime.now()), 'a', 1000000, 1)
# rfh.setFormatter(logging.Formatter(FILE_FORMAT))
# logger.addHandler(rfh)
# trace log (no recording)
tracer = logging.getLogger("Tracer")
# tracer.addHandler(handler)

# Log only for error
err_logger = logging.getLogger("Error")
# rfh1 = RotatingFileHandler('{:%Y-%m-%d %H%M%S}-Error.log'.format(datetime.now()),
# 'a', 1000000, 1)
# rfh1.setFormatter(logging.Formatter(FILE_FORMAT))
# err_logger.addHandler(rfh1)

OK = 1
FAILED = -1
NA = 0
DEBUG = 2


def trace(msg):
    tracer.info(msg)


def log(msg, success=NA):
    """
    log the result of a test, with colors depending on the state
    """

    if success is NA:
        logger.info(msg, extra={"state": "NA"})
    elif success is OK:
        logger.info(msg, extra={"state": "OK"})
    elif success is FAILED:
        logger.info(msg, extra={"state": "FAILED"})
    elif success is DEBUG:
        logger.info(msg, extra={"state": " "})

    if success is FAILED:
        err_logger.error(msg, extra={"state": "FAILED"})
