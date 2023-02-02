"""
This example runs an optical centering plan
"""

import time
from os import environ

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback

environ["BL_ACTIVE"] = "True"
environ["MD3_ADDRESS"] = "10.244.101.30"
environ["MD3_PORT"] = "9001"
environ["MD_REDIS_HOST"] = "10.244.101.30"
environ["MD_REDIS_PORT"] = "6379"
from mx3_beamline_library.devices.detectors import md_camera  # noqa
from mx3_beamline_library.devices.motors import md3  # noqa
from mx3_beamline_library.plans.optical_centering import OpticalCentering  # noqa

# Instantiate run engine and start plan
RE = RunEngine({})
bec = BestEffortCallback()
RE.subscribe(bec)

t = time.perf_counter()
optical_centering = OpticalCentering(
    camera=md_camera,
    sample_x=md3.sample_x,
    sample_y=md3.sample_y,
    alignment_x=md3.alignment_x,
    alignment_y=md3.alignment_y,
    alignment_z=md3.alignment_z,
    omega=md3.omega,
    zoom=md3.zoom,
    phase=md3.phase,
    backlight=md3.backlight,
    beam_position=[640, 512],
    auto_focus=True,
    min_focus=-0.3,
    max_focus=1.3,
    tol=0.3,
    number_of_intervals=2,
    plot=True,
    loop_img_processing_beamline="MX3",
    loop_img_processing_zoom="1",
    number_of_omega_steps=7,
)
RE(optical_centering.center_loop())

print(f"Execution time: {time.perf_counter() - t}")
