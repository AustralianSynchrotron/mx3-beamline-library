"""
This example runs a screening plan

    Requirements:
    - Access to the Dectris SIMPLON API (or Simulated SIMPLON-API)
    - Access to the MD3 exporter server. If the environment variable
        BL_ACTIVE=False, access to the server is not needed and ophyd
        simulated motors as used as a replacement.
    - A connection to a Redis server, which is used to store the
    beam center
"""

import time
from os import environ
from uuid import uuid4

import redis
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

# Instantiate run engine an start plan
RE = RunEngine({})
bec = BestEffortCallback()
RE.subscribe(bec)

# Mock beam center, assumes beam_center = a + b * distance + c * distance^2
redis_client = redis.StrictRedis()
redis_client.hset(
    name="beam_center_x_16M",
    mapping={
        "a": 2000.0,
        "b": 0.0,
        "c": 0.0,
    },
)
redis_client.hset(
    name="beam_center_y_16M",
    mapping={
        "a": 2000.0,
        "b": 0.0,
        "c": 0.0,
    },
)

# Run plan
t = time.perf_counter()
screening = md3_scan(
    acquisition_uuid=uuid4(),
    scan_range=20,
    exposure_time=2,
    number_of_frames=1,
    detector_distance=0.4,
    photon_energy=13,
    transmission=0.1,
    collection_type="screening",
)

RE(screening)

print(f"Execution time: {time.perf_counter() - t}")
