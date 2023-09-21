"""
This plan runs a 1D Scan. We manually simulate an Ophyd device that
simulates detector data. When we compute the derivative of this data
distribution, it follows a Gaussian distribution.
"""

from bluesky import RunEngine
from bluesky.plans import grid_scan, scan
from ophyd.sim import motor1, det1, motor2, motor3
import numpy as np
import os
os.environ["BL_ACTIVE"]="True"
os.environ["EPICS_CA_ADDR_LIST"] = "01.234.567.89 01.234.567.89" # Specify CA ADDRESS list here!
from mx3_beamline_library.devices.detectors import blackfly_camera
from mx3_beamline_library.devices.sim.classes.detectors import SimBlackFlyCam
import matplotlib.pyplot as plt
from ophyd.status import Status
from bluesky.plan_stubs import trigger_and_read
from ophyd import ADComponent, cam, AreaDetector, ImagePlugin, DeviceStatus, Signal, TransformPlugin, EpicsSignalWithRBV, Kind, SingleTrigger
from bluesky.plan_stubs import open_run, close_run
from collections import OrderedDict
from ophyd.areadetector.filestore_mixins import FileStoreHDF5IterativeWrite
import threading
from ophyd.areadetector.plugins import ColorConvPlugin, StatsPlugin_V33
from ophyd.areadetector.filestore_mixins import FileStoreIterativeWrite
from ophyd.areadetector.plugins import HDF5Plugin
from os import getcwd
from ophyd.areadetector.base import DDC_EpicsSignalRO
from bluesky.plan_stubs import mv
from time import sleep
from ophyd import Component as Cpt
from ophyd.status import Status
from mx3_beamline_library.devices.classes.detectors import GrasshopperCamera, HDF5Filewriter
from toolz import partition
from bluesky.utils import Msg, merge_cycler
from cycler import cycler
from functools import reduce
import operator 
import uuid
from mx3_beamline_library.plans.commissioning.commissioning import Scan1D
from scipy.stats import skewnorm
from ophyd.signal import EpicsSignalRO

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt

def smoothstep(x, x_min=0, x_max=1, N=1,flip=False):
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
INTENSITY_SIM = smoothstep(x, N=5,flip=True)
print(INTENSITY_SIM)


class MySignal(EpicsSignalRO):
    def __init__(self, read_pv, *, string=False, name=None, **kwargs):
        super().__init__(read_pv, string=string, name=name, **kwargs)
        self.counter = 0
    
    def get(self):
        value = INTENSITY_SIM[self.counter]
        self.counter +=1
        return value
    
class MyStats(StatsPlugin_V33):
    total = ADComponent(MySignal, "Total_RBV")
    
class SimGrasshopper(GrasshopperCamera):
    stats = ADComponent(MyStats, ":" + StatsPlugin_V33._default_suffix)
    


my_cam = SimGrasshopper("13SIM1", name="blackfly_camera")


motor1.delay = 0.001

RE = RunEngine({})
print(my_cam.stats.total.get())

scan_1d = Scan1D([my_cam.stats.total],
        motor1, -1,1, 50,
        hdf5_filename=None, dwell_time=10, calculate_first_derivative=True)
RE(scan_1d.run())
print(scan_1d.statistics)


