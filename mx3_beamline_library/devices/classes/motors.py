""" Motor Definitions """

from as_acquisition_library.devices.motors import ASEpicsMotor
from ophyd import Component as Cpt, MotorBundle


class MyTable(MotorBundle):
    """A Generic Table."""

    x = Cpt(ASEpicsMotor, name="AXIS:X")
    y = Cpt(ASEpicsMotor, name="AXIS:Y")
    z = Cpt(ASEpicsMotor, name="AXIS:Z")
