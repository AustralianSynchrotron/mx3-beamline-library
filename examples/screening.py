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
environ["DECTRIS_DETECTOR_HOST"] = "12.345.678.90"
environ["DECTRIS_DETECTOR_PORT"] = "80"
environ["MD_REDIS_HOST"] = "12.345.678.90"
environ["MD_REDIS_PORT"] = "6379"
environ["MD3_ADDRESS"] = "12.345.678.90"
environ["MD3_PORT"] = "9001"
from mx3_beamline_library.devices import detectors  # noqa
from mx3_beamline_library.plans.basic_scans import md3_scan  # noqa
from mx3_beamline_library.schemas.crystal_finder import MotorCoordinates  # noqa

dectris_detector = detectors.dectris_detector


# Instantiate run engine an start plan
RE = RunEngine({})
bec = BestEffortCallback()
RE.subscribe(bec)

t = time.perf_counter()
screening = md3_scan(
    id="my_sample",
    motor_positions=MotorCoordinates(
        sample_x=0,
        sample_y=0,
        alignment_x=0.434,
        alignment_y=0,
        alignment_z=0,
        omega=0,
    ),
    number_of_frames=10,
    scan_range=10,
    exposure_time=1,
)
RE(screening)

print(f"Execution time: {time.perf_counter() - t}")
