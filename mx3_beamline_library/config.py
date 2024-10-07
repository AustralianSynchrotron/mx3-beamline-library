from os import environ, path

import yaml
from redis import StrictRedis
from redis.exceptions import ConnectionError

from .logger import setup_logger

logger = setup_logger()

# Determine which mode the beamline library is running on, by default it is run
# in SIM mode
BL_ACTIVE = environ.get("BL_ACTIVE", "false").lower()

# Redis connection
REDIS_HOST = environ.get("REDIS_HOST", "0.0.0.0")
REDIS_PORT = int(environ.get("REDIS_PORT", "6379"))
REDIS_USERNAME = environ.get("REDIS_USERNAME", None)
REDIS_PASSWORD = environ.get("REDIS_PASSWORD", None)
REDIS_DB = int(environ.get("REDIS_DB", "0"))

try:
    redis_connection = StrictRedis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        username=REDIS_USERNAME,
        password=REDIS_PASSWORD,
        db=REDIS_DB,
    )
except ConnectionError:
    logger.warning(
        "A redis connection is not available. Some functionalities may be limited."
    )

with open(
    path.join(path.dirname(__file__), "devices", "classes", "md3_config.yml")
) as config:
    MD3_CONFIG = yaml.safe_load(config)
