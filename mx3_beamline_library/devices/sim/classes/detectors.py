from ophyd import Component as Cpt, Device
from ophyd.signal import EpicsSignalRO
from ophyd.sim import DetWithCountTime


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


class MySimDetector(DetWithCountTime):
    """A simulated detector"""
