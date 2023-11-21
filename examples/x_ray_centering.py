"""
This example runs a grid scan on a sample. The coordinates for the grid scan come from redis
and are obtained by running the optical_centering plan (see the optical_centering.py example).
Before running this example, make sure to run the optical_centering plan first.

    Requirements:
    - A Redis connection
    - Access to the Dectris SIMPLON API (or Simulated SIMPLON-API)

    Optional requirements:
    - Access to the MD3 exporter server. If the environment variable
    BL_ACTIVE=False, access to the server is not needed and ophyd
    simulated motors as used as a replacement.
"""
import time
from os import environ

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback

# Modify the following ENV variables with the corresponding
# hosts and ports.
# IF BL_ACTIVE=False, we run the library in simulation mode
environ["BL_ACTIVE"] = "False"
environ["SIMPLON_API"] = "http://0.0.0.0:8000"
environ["MD_REDIS_HOST"] = "12.345.678.90"
environ["MD_REDIS_PORT"] = "6379"
environ["MD3_ADDRESS"] = "12.345.678.90"
environ["MD3_PORT"] = "9001"
from mx3_beamline_library.plans.xray_centering import XRayCentering  # noqa

# Instantiate run engine an start plan
RE = RunEngine({})
bec = BestEffortCallback()
RE.subscribe(bec)

t = time.perf_counter()
xray_centering = XRayCentering(
    sample_id="my_sample",
    grid_scan_id="flat",
    exposure_time=1,
)
RE(xray_centering.start_grid_scan())

print(f"Execution time: {time.perf_counter() - t}")
