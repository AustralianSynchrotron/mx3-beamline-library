""" Beamline detector definition """

from ophyd.areadetector import DetectorBase
from ophyd import EpicsSignal, Component as Cpt

class MyDetector(DetectorBase):
    """ An pathalogical example of a custom detector. """
    file_name = Cpt(EpicsSignal, "cam1:FileName_RBV", write_pv="cam1:FileName", string=True)
    file_path = Cpt(EpicsSignal, "cam1:FilePath_RBV", write_pv="cam1:FilePath", string=True)
