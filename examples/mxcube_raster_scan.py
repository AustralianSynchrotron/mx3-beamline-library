"""
This example runs a grid scan on a sample based on parameters obtained from mxcube.

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
environ["MD3_REDIS_HOST"] = "12.345.678.90"
environ["MD3_REDIS_PORT"] = "6379"
environ["MD3_ADDRESS"] = "12.345.678.90"
environ["MD3_PORT"] = "9001"
from mx3_beamline_library.plans.manual_xray_centering import ManualXRayCentering  # noqa

# Instantiate run engine an start plan
RE = RunEngine({})
bec = BestEffortCallback()
RE.subscribe(bec)

t = time.perf_counter()
xray_centering = ManualXRayCentering(
    sample_id="my_sample",
    grid_scan_id="manual_collection",
    grid_top_left_coordinate=(388, 502),
    grid_height=260,
    grid_width=364,
    beam_position=(612, 512),
    number_of_columns=7,
    number_of_rows=5,
    detector_distance=0.496,  # m
    photon_energy=13,  # keV
    transmission=0.1,
    omega_range=0,  # degrees
    md3_alignment_y_speed=1,  # mm/s
    count_time=None,
    hardware_trigger=True,
)
RE(xray_centering.start_grid_scan())

print(f"Execution time: {time.perf_counter() - t}")
