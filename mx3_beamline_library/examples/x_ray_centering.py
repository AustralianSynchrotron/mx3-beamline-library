"""
This example runs an optical and xray centering plan. To run this example,
you should have a running instance of the sim-plon API. For details on how to run the
sim-plon api, refer to https://bitbucket.synchrotron.org.au/scm/mx3/mx-sim-plon-api.git,
and make sure that the DECTRIS_DETECTOR_HOST and DECTRIS_DETECTOR_PORT (see below) are
configured accordingly.
"""
import time
from os import environ

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback

environ["DECTRIS_DETECTOR_HOST"] = "12.345.678.90"
environ["DECTRIS_DETECTOR_PORT"] = "80"
environ["BL_ACTIVE"] = "True"
environ["MD_REDIS_HOST"] = "12.345.678.90"
environ["MD_REDIS_PORT"] = "6379"
environ["MD3_ADDRESS"] = "12.345.678.90"
environ["MD3_PORT"] = "9001"
from mx3_beamline_library.devices import detectors, motors  # noqa
from mx3_beamline_library.devices.detectors import md_camera  # noqa
from mx3_beamline_library.devices.motors import md3  # noqa
from mx3_beamline_library.plans.xray_centering import XRayCentering  # noqa

# Configure the detector to send one frame per trigger
# REST = "http://0.0.0.0:8000"
# nimages = {"value": 1}
# r = requests.put(f"{REST}/detector/api/1.8.0/config/nimages", json=nimages)

dectris_detector = detectors.dectris_detector


# Instantiate run engine an start plan
RE = RunEngine({})
bec = BestEffortCallback()
RE.subscribe(bec)
print(md3.phase.get())

t = time.perf_counter()
xray_centering = XRayCentering(
    sample_id="my_sample",
    detector=dectris_detector,
    omega=md3.omega,
    zoom=md3.zoom,
    grid_scan_id="flat",
)
RE(xray_centering.start_grid_scan())

print(f"Execution time: {time.perf_counter() - t}")
