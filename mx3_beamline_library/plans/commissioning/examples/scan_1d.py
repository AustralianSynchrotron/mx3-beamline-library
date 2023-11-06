"""
This plan runs a 1D Scan. We manually simulate an Ophyd device that
generates detector data following a Gaussian distribution
"""

import os

import numpy as np
from bluesky import RunEngine
from ophyd import ADComponent
from ophyd.areadetector.plugins import StatsPlugin_V33
from ophyd.signal import EpicsSignalRO
from ophyd.sim import motor1
from scipy.stats import skewnorm

os.environ["BL_ACTIVE"] = "True"
# Specify CA ADDRESS list here!
os.environ[
    "EPICS_CA_ADDR_LIST"
] = "01.234.567.89 01.234.567.89"  # Specify CA ADDRESS list here!
from mx3_beamline_library.devices.classes.detectors import BlackflyCamera  # noqa
from mx3_beamline_library.plans.commissioning.commissioning import Scan1D  # noqa

x = np.linspace(0, 100, 54)
a = -9
INTENSITY_SIM = 100 * skewnorm.pdf(x, a, loc=50, scale=9.8)


class MySignal(EpicsSignalRO):
    def __init__(self, read_pv, *, string=False, name=None, **kwargs):
        super().__init__(read_pv, string=string, name=name, **kwargs)
        self.counter = 0

    def get(self):
        value = INTENSITY_SIM[self.counter]
        self.counter += 1
        return value


class MyStats(StatsPlugin_V33):
    total = ADComponent(MySignal, "Total_RBV")


class SimGrasshopper(BlackflyCamera):
    stats = ADComponent(MyStats, ":" + StatsPlugin_V33._default_suffix)


my_cam = SimGrasshopper("13SIM1", name="blackfly_camera")

motor1.delay = 0.001

RE = RunEngine({})
print(my_cam.stats.total.get())

scan_1d = Scan1D(
    [my_cam.stats.total],
    motor1,
    -1,
    100,
    50,
    hdf5_filename=None,
    dwell_time=10,
    calculate_first_derivative=False,
)
RE(scan_1d.run())
print(scan_1d.statistics)
