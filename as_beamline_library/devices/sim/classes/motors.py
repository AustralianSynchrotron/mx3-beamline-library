""" Simulated motor definitions specific to this beamline. """

from as_acquisition_library.devices.motors import ASSimMotor
from ophyd import Component as Cpt, MotorBundle


class MySimTable(MotorBundle):
    """A Simulated Generic Table."""

    x = Cpt(ASSimMotor, name="AXIS:X")
    y = Cpt(ASSimMotor, name="AXIS:Y")
    z = Cpt(ASSimMotor, name="AXIS:Z")
