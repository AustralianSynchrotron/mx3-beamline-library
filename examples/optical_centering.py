"""
This example aligns the tip of the loop with the center of the beam,
finds the angles corresponding to the loop's maximum and minimum area,
and infers the coordinates needed to execute rasters scans on the loop.
The results are saved to redis

    Requirements:
    - A Redis connection

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
environ["MD3_ADDRESS"] = "12.345.678.90"
environ["MD3_PORT"] = "9001"
environ["MD3_REDIS_HOST"] = "12.345.678.90"
environ["MD3_REDIS_PORT"] = "6379"
from mx3_beamline_library.plans.optical_centering import OpticalCentering  # noqa

# Instantiate run engine and start plan
RE = RunEngine({})
bec = BestEffortCallback()
RE.subscribe(bec)

t = time.perf_counter()
optical_centering = OpticalCentering(
    sample_id="my_sample",
    beam_position=(612, 512),
    grid_step=(100, 100),
    plot=True,
    calibrated_alignment_z=0.45,
    manual_mode=False,
    use_top_camera_camera=True,
)
RE(optical_centering.center_loop())

print(f"Execution time: {time.perf_counter() - t}")
