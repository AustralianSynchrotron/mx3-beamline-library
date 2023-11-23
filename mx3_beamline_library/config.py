import logging
from os import environ, path

from redis import StrictRedis
from redis.exceptions import ConnectionError
from yaml import safe_load


# Logger config
def setup_logger():
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a console handler and set the level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and attach it to the console handler
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()

# Determine which mode the beamline library is running on, by default it is run
# in SIM mode
BL_ACTIVE = environ.get("BL_ACTIVE", "false").lower()

# Plan configuration files
path_to_config_file = path.join(
    path.dirname(__file__), "./plans/configuration/optical_centering.yml"
)
with open(path_to_config_file, "r") as plan_config:
    OPTICAL_CENTERING_CONFIG: dict = safe_load(plan_config)

# Redis connection
REDIS_HOST = environ.get("REDIS_HOST", "0.0.0.0")
REDIS_PORT = int(environ.get("REDIS_PORT", "6379"))
try:
    redis_connection = StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)
except ConnectionError:
    logger.warning(
        "A redis connection is not available. Some functionalities may be limited."
    )
