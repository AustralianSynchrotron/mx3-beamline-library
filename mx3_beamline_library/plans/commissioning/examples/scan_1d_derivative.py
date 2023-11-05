"""
This plan runs a 1D Scan. We manually simulate an Ophyd device that
simulates detector data. When we compute the derivative of this data
distribution, it follows a Gaussian distribution.
"""

import os

import numpy as np
from bluesky import RunEngine
from ophyd import ADComponent
from ophyd.areadetector.plugins import StatsPlugin_V33
from ophyd.signal import EpicsSignalRO
from ophyd.sim import motor1
from scipy.special import comb

os.environ["BL_ACTIVE"] = "True"
os.environ[
    "EPICS_CA_ADDR_LIST"
] = "01.234.567.89 01.234.567.89"  # Specify CA ADDRESS list here!
from mx3_beamline_library.devices.classes.detectors import BlackflyCamera  # noqa
from mx3_beamline_library.plans.commissioning.commissioning import Scan1D  # noqa


def smoothstep(x, x_min=0, x_max=1, N=1, flip=False):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)
    if flip:
        return np.flipud(result)
    else:
        return result


x = np.linspace(0, 1, 54)
INTENSITY_SIM = smoothstep(x, N=5, flip=True)
print(INTENSITY_SIM)


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
    1,
    50,
    hdf5_filename=None,
    dwell_time=10,
    calculate_first_derivative=True,
)
RE(scan_1d.run())
print(scan_1d.statistics)
