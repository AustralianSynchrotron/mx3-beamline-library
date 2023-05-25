
from os import environ

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback

environ["BL_ACTIVE"] = "True"
environ["MD3_ADDRESS"] = "10.244.101.30"
environ["MD3_PORT"] = "9001"
environ["MD_REDIS_HOST"] = "10.244.101.30"
environ["MD_REDIS_PORT"] = "6379"

from mx3_beamline_library.plans.tray_scans import get_drop_images

# Instantiate run engine an start plan
RE = RunEngine({})
bec = BestEffortCallback()
RE.subscribe(bec)

drop_locations = ["B1-1"]
RE(get_drop_images(drop_locations, alignment_y_offset=0.25, alignment_z_offset=-1.0,))