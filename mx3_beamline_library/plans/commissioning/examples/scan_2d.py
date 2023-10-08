"""
This example shows how to run a 2D scan
"""

import os

from bluesky import RunEngine
from ophyd.sim import motor1, motor2

os.environ["BL_ACTIVE"] = "True"
os.environ[
    "EPICS_CA_ADDR_LIST"
] = "01.234.567.89 01.234.567.89"  # Specify CA ADDRESS list here!
from mx3_beamline_library.devices.classes.detectors import GrasshopperCamera  # noqa
from mx3_beamline_library.plans.commissioning.commissioning import Scan2D  # noqa

my_cam = GrasshopperCamera("13SIM1", name="blackfly_camera")

motor1.delay = 0.001
motor2.delay = 0.001


RE = RunEngine({})
print(my_cam.stats.total.get())

scan_2d = Scan2D([my_cam.stats.total], motor1, -1, 100, 5, motor2, -1, 100, 5)
RE(scan_2d.run())
