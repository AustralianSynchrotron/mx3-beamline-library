"""
This example runs an optical and xray centering plan. To run this example,
you should have a running instance of the sim-plon API. For details on how to run the
sim-plon api, refer to https://bitbucket.synchrotron.org.au/scm/mx3/mx-sim-plon-api.git,
and make sure that the DECTRIS_DETECTOR_HOST and DECTRIS_DETECTOR_PORT (see below) are
configured accordingly.
"""
from os import environ
import requests

environ["DECTRIS_DETECTOR_HOST"]= "0.0.0.0"
environ["DECTRIS_DETECTOR_PORT"]= "8000"
environ["BL_ACTIVE"] = "True"
environ["SETTLE_TIME"] = "0.2"


from mx3_beamline_library.plans.optical_and_xray_centering import OpticalAndXRayCentering, optical_and_xray_centering
from mx3_beamline_library.devices import detectors, motors
from bluesky import RunEngine
from ophyd.sim import det
from bluesky.plan_stubs import mv
from bluesky.callbacks.best_effort import BestEffortCallback


# Configure the detector to send one frame per trigger
REST = "http://0.0.0.0:8000"
nimages = {"value": 1}
r = requests.put(f"{REST}/detector/api/1.8.0/config/nimages", json=nimages)

# Instantiate devices
camera = detectors.blackfly_camera
testrig = motors.testrig
motor_x = testrig.x
motor_x.wait_for_connection()
motor_z = testrig.z
motor_z.wait_for_connection()
motor_y = testrig.y
motor_y.wait_for_connection()
motor_phi = testrig.phi
motor_phi.wait_for_connection()
dectris_detector = detectors.dectris_detector

# Drive motors to a position where we can see the loop
motor_z.move(0, wait=True)
motor_x.move(0, wait=True)


# Instantiate run engine an start plan
RE = RunEngine({})
bec = BestEffortCallback()
RE.subscribe(bec)

optical_and_xray_centering = OpticalAndXRayCentering(
    dectris_detector, 
    camera, 
    motor_x,
    10,
    motor_y,
    motor_z,
    2,
    motor_phi,
    md={"sample_id": "my_test_sample"},
    beam_position=[640, 512],
    pixels_per_mm_x=292.87,
    pixels_per_mm_z=292.87,
    threshold=20,
    auto_focus=False,
    min_focus=-1,
    max_focus=0.,
    tol=0.1,
    method="psi",
    plot=True
)
RE(optical_and_xray_centering.start())

