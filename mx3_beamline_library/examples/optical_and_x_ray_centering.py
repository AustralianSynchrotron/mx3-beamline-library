"""
This example runs an optical and xray centering plan. To run this example,
you should have a running instance of the sim-plon API. For details on how to run the
sim-plon api, refer to https://bitbucket.synchrotron.org.au/scm/mx3/mx-sim-plon-api.git,
and make sure that the DECTRIS_DETECTOR_HOST and DECTRIS_DETECTOR_PORT (see below) are
configured accordingly.
"""
from os import environ

import requests
from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback

environ["DECTRIS_DETECTOR_HOST"] = "0.0.0.0"
environ["DECTRIS_DETECTOR_PORT"] = "8000"
environ["BL_ACTIVE"] = "True"
environ["MD_REDIS_HOST"] = "10.244.101.30"
environ["MD_REDIS_PORT"] = "6379"
from mx3_beamline_library.devices import detectors, motors  # noqa
from mx3_beamline_library.plans.optical_and_xray_centering import (  # noqa
    OpticalAndXRayCentering,
)
from mx3_beamline_library.devices.motors import md3  # noqa
from mx3_beamline_library.devices.detectors import md_camera  # noqa

# Configure the detector to send one frame per trigger
REST = "http://0.0.0.0:8000"
nimages = {"value": 1}
r = requests.put(f"{REST}/detector/api/1.8.0/config/nimages", json=nimages)

dectris_detector = detectors.dectris_detector


# Instantiate run engine an start plan
RE = RunEngine({})
bec = BestEffortCallback()
RE.subscribe(bec)
print(md3.phase.get())

optical_and_xray_centering = OpticalAndXRayCentering(
    detector=dectris_detector,
    camera=md_camera,
    sample_x=md3.sample_x,
    sample_y=md3.sample_y,
    alignment_x=md3.alignment_x,
    alignment_y=md3.alignment_y,
    alignment_z=md3.alignment_z,
    omega=md3.omega,
    zoom=md3.zoom,
    phase=md3.phase,
    beam_position=[640, 512],
    auto_focus=True,
    min_focus=0,
    max_focus=1.3,
    tol=0.5,
    number_of_intervals=2,
    plot=True,
    loop_img_processing_beamline="MD3",
    loop_img_processing_zoom="1",
    beam_size=(100, 100),
    md={"sample_id": "sample_test"}
)
RE(optical_and_xray_centering.start())
