""" Motor Definitions """

from ophyd import MotorBundle, Component as Cpt
from as_acquisition_library.devices.motors import ASEpicsMotor

class MyTable(MotorBundle):
    """A Generic Table."""

    x = Cpt(ASEpicsMotor, name="AXIS:X")
    y = Cpt(ASEpicsMotor, name="AXIS:Y")
    z = Cpt(ASEpicsMotor, name="AXIS:Z")

