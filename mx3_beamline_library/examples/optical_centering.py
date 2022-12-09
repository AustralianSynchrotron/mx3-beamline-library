"""
This example runs an optical centering plan
"""

from os import environ


environ["BL_ACTIVE"] = "True"
environ["SETTLE_TIME"] = "0.2"


from mx3_beamline_library.plans.optical_centering import OpticalCentering
from mx3_beamline_library.devices import detectors, motors
from bluesky import RunEngine
from bluesky.plan_stubs import mv
from bluesky.callbacks.best_effort import BestEffortCallback


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

# Instantiate run engine an start plan
RE = RunEngine({})
bec = BestEffortCallback()
RE.subscribe(bec)
RE(mv(motor_x, 0, motor_z, 0, motor_y, -0.84))

optical_centering = OpticalCentering(
    camera, 
    motor_x, 
    motor_y,
    motor_z,
    motor_phi,
    beam_position=[640, 512],
    pixels_per_mm_x=292.87,
    pixels_per_mm_z=292.87,
    auto_focus=True,
    min_focus=-1,
    max_focus=0.5,
    tol=0.1,
    method="psi",
    plot=False,)
RE(optical_centering.center_loop())


