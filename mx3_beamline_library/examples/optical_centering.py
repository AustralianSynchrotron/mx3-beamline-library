"""
This example runs an optical centering plan
"""

from os import environ

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.plan_stubs import mv

environ["BL_ACTIVE"] = "True"
environ["MD3_ADDRESS"] = "10.244.101.30"
environ["MD3_PORT"] = "9001"
environ["MD_REDIS_HOST"] = "10.244.101.30"
environ["MD_REDIS_PORT"] = "6379"
from mx3_beamline_library.devices.motors import md3  # noqa
from mx3_beamline_library.devices.detectors import md_camera #noqa
from mx3_beamline_library.plans.optical_centering import OpticalCentering


# Instantiate run engine an start plan
RE = RunEngine({})
bec = BestEffortCallback()
RE.subscribe(bec)
RE(mv(md3.sample_x, 0, md3.sample_y, 1, md3.alignment_y, 0))

optical_centering = OpticalCentering(
    md_camera,
    md3.sample_x,
    md3.sample_y,
    md3.alignment_y,
    md3.omega,
    beam_position=[640, 512],
    pixels_per_mm_x=520.97,
    pixels_per_mm_z=520.97,
    auto_focus=True,
    min_focus=-0.5,
    max_focus=0,
    tol=0.1,
    method="psi",
    plot=True,
)
RE(optical_centering.center_loop())
