from os import environ, path

from yaml import safe_load

BL_ACTIVE = environ.get("BL_ACTIVE", "false").lower()

path_to_config_file = path.join(
    path.dirname(__file__), "./plans/configuration/optical_centering.yml"
)
with open(path_to_config_file, "r") as plan_config:
    OPTICAL_CENTERING_CONFIG: dict = safe_load(plan_config)
