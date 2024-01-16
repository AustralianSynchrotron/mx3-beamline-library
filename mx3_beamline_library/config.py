from os import environ

from redis import StrictRedis
from redis.exceptions import ConnectionError

from .logger import setup_logger
from .schemas.optical_centering import OpticalCenteringExtraConfig

logger = setup_logger()

# Determine which mode the beamline library is running on, by default it is run
# in SIM mode
BL_ACTIVE = environ.get("BL_ACTIVE", "false").lower()

# Plan configuration files
OPTICAL_CENTERING_CONFIG = OpticalCenteringExtraConfig()

# Redis connection
REDIS_HOST = environ.get("REDIS_HOST", "0.0.0.0")
REDIS_PORT = int(environ.get("REDIS_PORT", "6379"))
try:
    redis_connection = StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)
except ConnectionError:
    logger.warning(
        "A redis connection is not available. Some functionalities may be limited."
    )
