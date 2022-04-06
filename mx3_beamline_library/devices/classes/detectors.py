""" Beamline detector definition """

from ophyd import Component as Cpt, EpicsSignal
from ophyd.areadetector import CamBase, DetectorBase


class MyDetector(DetectorBase):
    """An pathalogical example of a custom detector."""

    cam = Cpt(CamBase)
    file_name = Cpt(
        EpicsSignal, "cam1:FileName_RBV", write_pv="cam1:FileName", string=True
    )
    file_path = Cpt(
        EpicsSignal, "cam1:FilePath_RBV", write_pv="cam1:FilePath", string=True
    )
