""" Motor Definitions """

from ophyd import Component as Cpt, EpicsMotor, MotorBundle


class Testrig(MotorBundle):
    """
    Testrig motors
    """

    x = Cpt(EpicsMotor, ":mtr2", lazy=True)
    y = Cpt(EpicsMotor, ":mtr3", lazy=True)
    z = Cpt(EpicsMotor, ":mtr1", lazy=True)


class MxcubeSimulatedPVs(MotorBundle):
    """
    Simulated PVs for Mxcube
    """

    m1 = Cpt(EpicsMotor, ":m1", lazy=True)
    m2 = Cpt(EpicsMotor, ":m2", lazy=True)
    m3 = Cpt(EpicsMotor, ":m3", lazy=True)
    m4 = Cpt(EpicsMotor, ":m4", lazy=True)
    m5 = Cpt(EpicsMotor, ":m5", lazy=True)
    m6 = Cpt(EpicsMotor, ":m6", lazy=True)
    m7 = Cpt(EpicsMotor, ":m7", lazy=True)
    m8 = Cpt(EpicsMotor, ":m8", lazy=True)
