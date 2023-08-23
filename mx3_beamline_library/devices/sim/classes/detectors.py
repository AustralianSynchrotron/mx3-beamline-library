from os import path

import numpy as np
import numpy.typing as npt
from ophyd import Component as Cpt, Device
from ophyd.signal import EpicsSignalRO, Signal
from ophyd.sim import DetWithCountTime
from ophyd.areadetector.cam import SimDetectorCam
from ...classes.detectors import GrasshopperCamera
from . import Register

path_to_sim_images = path.join(path.dirname(__file__), "../sim_images")

SIM_TOP_CAMERA_IMG = np.load(path.join(path_to_sim_images, "top_camera.npy"))
SIM_MD3_CAMERA_IMG = np.load(path.join(path_to_sim_images, "md3_image.npy"))

@Register("Grasshopper Camera")
class GrasshopperCamera(SimDetectorCam):
    pass


class BlackFlyCam(Device):
    """Ophyd device to acquire images from a simulated Blackfly camera

    Attributes
    ----------
    depth: float
        Depth of the camera image
    width: float
        Width of the camera image
    height: float
        Height of the camera image
    array_data : numpy array
        Array data
    """

    depth = Cpt(EpicsSignalRO, ":image1:ArraySize0_RBV")
    width = Cpt(EpicsSignalRO, ":image1:ArraySize1_RBV")
    height = Cpt(EpicsSignalRO, ":image1:ArraySize2_RBV")
    array_data = Cpt(EpicsSignalRO, ":image1:ArrayData")


class SimBlackFlyCam(Device):
    """Ophyd device to acquire images from a simulated Blackfly camera

    Attributes
    ----------
    depth: float
        Depth of the camera image
    width: float
        Width of the camera image
    height: float
        Height of the camera image
    array_data : numpy array
        Array data
    """

    depth = Cpt(Signal, kind="hinted", value=0)
    width = Cpt(Signal, kind="hinted", value=0)
    height = Cpt(Signal, kind="hinted", value=0)
    data = SIM_MD3_CAMERA_IMG
    array_data = Cpt(Signal, kind="hinted", value=data)

    pixels_per_mm_x = 50
    pixels_per_mm_y = 50

    def set_values(self, snapshot: npt.NDArray) -> None:
        """ """

        width: int = 0
        height: int = 0
        depth: int = 0

        try:
            height, width, depth = snapshot.shape
        except ValueError:
            height, width = snapshot.shape
        self.array_data.set(snapshot)
        self.width.set(width)
        self.height.set(height)
        self.depth.set(depth)


class MySimDetector(DetWithCountTime):
    """A simulated detector"""
