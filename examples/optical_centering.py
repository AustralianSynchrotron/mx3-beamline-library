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
environ["MD_REDIS_HOST"] = "12.345.678.90"
environ["MD_REDIS_PORT"] = "6379"
from mx3_beamline_library.devices.detectors import blackfly_camera, md_camera  # noqa
from mx3_beamline_library.devices.motors import md3  # noqa
from mx3_beamline_library.plans.optical_centering import optical_centering  # noqa

# Instantiate run engine and start plan
RE = RunEngine({})
bec = BestEffortCallback()
RE.subscribe(bec)

t = time.perf_counter()
_optical_centering = optical_centering(
    sample_id="my_sample",
    md3_camera=md_camera,
    top_camera=blackfly_camera,
    sample_x=md3.sample_x,
    sample_y=md3.sample_y,
    alignment_x=md3.alignment_x,
    alignment_y=md3.alignment_y,
    alignment_z=md3.alignment_z,
    omega=md3.omega,
    zoom=md3.zoom,
    phase=md3.phase,
    backlight=md3.backlight,
    beam_position=(640, 512),
    grid_step=(80, 80),
)
RE(_optical_centering)

print(f"Execution time: {time.perf_counter() - t}")