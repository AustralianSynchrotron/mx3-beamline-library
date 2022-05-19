""" Simulated motor definitions specific to this beamline. """

from as_acquisition_library.devices.motors import ASSimMotor
from ophyd import Component as Cpt, MotorBundle


class MySimTable(MotorBundle):
    """A Simulated Generic Table."""

    x = Cpt(ASSimMotor, name="AXIS:X")
    y = Cpt(ASSimMotor, name="AXIS:Y")
    z = Cpt(ASSimMotor, name="AXIS:Z")


class SimulatedPVs(MotorBundle):
    """
    Simulated PVs for Mxcube
    """

    m1 = Cpt(ASSimMotor, name="MXCUBE:m1")
    m2 = Cpt(ASSimMotor, name="MXCUBE:m2")
    m3 = Cpt(ASSimMotor, name="MXCUBE:m3")
    m4 = Cpt(ASSimMotor, name="MXCUBE:m4")
    m5 = Cpt(ASSimMotor, name="MXCUBE:m5")
    m6 = Cpt(ASSimMotor, name="MXCUBE:m6")
    m7 = Cpt(ASSimMotor, name="MXCUBE:m7")
    m8 = Cpt(ASSimMotor, name="MXCUBE:m8")
