"""
This plan runs a 1D Scan. We manually simulate an Ophyd device that
generates detector data following a Gaussian distribution
"""

from bluesky import RunEngine
from bluesky.plans import grid_scan, scan
from ophyd.sim import motor1, det1, motor2, motor3
import numpy as np
import os
os.environ["BL_ACTIVE"]="True"
# Specify CA ADDRESS list here!
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

x = np.linspace(0, 100, 54)
a = -9
INTENSITY_SIM = 100*skewnorm.pdf(x, a, loc=50, scale=9.8)

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
        motor1, -1,100, 50,
        hdf5_filename=None, dwell_time=10, calculate_first_derivative=False)
RE(scan_1d.run())
print(scan_1d.statistics)



