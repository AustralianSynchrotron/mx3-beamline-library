from os import environ, path

from redis import StrictRedis
from redis.exceptions import ConnectionError
from yaml import safe_load

from .logger import setup_logger

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
