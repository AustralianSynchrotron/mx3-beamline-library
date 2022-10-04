from os import environ  # noqa

environ["DECTRIS_DETECTOR_HOST"] = "0.0.0.0"  # noqa
environ["DECTRIS_DETECTOR_PORT"] = "8000"  # noqa
environ["BL_ACTIVE"] = "True"  # noqa
environ["BLUESKY_DEBUG_CALLBACKS"] = "1"  # noqa
environ["SETTLE_TIME"] = "0.2"  # noqa

import logging

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.plan_stubs import mv

from mx3_beamline_library.devices import detectors, motors
from mx3_beamline_library.plans.optical_and_xray_centering import (
    optical_and_xray_centering,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)


# Instantiate ophyd devices
testrig = motors.testrig
motor_x = testrig.x
motor_x.wait_for_connection()
motor_z = testrig.z
motor_z.wait_for_connection()
motor_y = testrig.y
motor_y.wait_for_connection()
motor_phi = testrig.phi
motor_phi.wait_for_connection()

blackfly_camera = detectors.blackfly_camera
blackfly_camera.wait_for_connection()

dectris_detector = detectors.dectris_detector

# Create the bluesky Run Engine
bec = BestEffortCallback()
RE = RunEngine({})
RE.subscribe(bec)

# We will need a low-resolution pin centering to get the loop in the camera view,
# but for now we move the motors to a position where we can see the loop
RE(mv(motor_x, 0, motor_z, 0, motor_phi, 0))

logging.info("Starting bluesky plan")

RE(
    optical_and_xray_centering(
        dectris_detector,
        motor_x,
        motor_z,
        motor_phi,
        blackfly_camera,
        md={"sample_id": "test_sample"},
        plot=False,
    )
)
