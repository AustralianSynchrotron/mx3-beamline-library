""" Simulated motor definitions specific to this beamline. """

from ophyd import MotorBundle, Component as Cpt
from as_acquisition_library.devices.motors import ASSimMotor

class MySimTable(MotorBundle):
    """A Simulated Generic Table."""

    x = Cpt(ASSimMotor, name="AXIS:X")
    y = Cpt(ASSimMotor, name="AXIS:Y")
    z = Cpt(ASSimMotor, name="AXIS:Z")
