from ophyd import Component as Cpt, Device
from ophyd.signal import EpicsSignalRO,  Signal
from ophyd.sim import DetWithCountTime
import numpy.typing as npt
import numpy as np


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
    data = np.load("/mnt/shares/smd_share/blackfly_cam_images/flat.py")
    array_data = Cpt(Signal, kind="hinted", value=data)
    

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
