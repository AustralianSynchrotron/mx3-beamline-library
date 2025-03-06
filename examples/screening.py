"""
This example runs a screening plan

    Requirements:
    - Access to the Dectris SIMPLON API (or Simulated SIMPLON-API)
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
environ["MD3_REDIS_HOST"] = "12.345.678.90"
environ["MD3_REDIS_PORT"] = "6379"
environ["MD3_ADDRESS"] = "12.345.678.90"
environ["MD3_PORT"] = "9001"
from mx3_beamline_library.plans.basic_scans import md3_scan  # noqa
from mx3_beamline_library.schemas.crystal_finder import MotorCoordinates  # noqa

# Instantiate run engine an start plan
RE = RunEngine({})
bec = BestEffortCallback()
RE.subscribe(bec)

t = time.perf_counter()
screening = md3_scan(
    id="my_sample",
    scan_range=20,
    exposure_time=2,
    number_of_frames=200,
    detector_distance=0.3,
    photon_energy=13,
    transmission=0.1,
)

RE(screening)

print(f"Execution time: {time.perf_counter() - t}")
