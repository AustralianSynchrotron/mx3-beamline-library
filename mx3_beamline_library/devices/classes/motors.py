""" Motor Definitions """

from ophyd import Component as Cpt, MotorBundle, EpicsMotor


class Testrig(MotorBundle):
    """
    Testrig motors
    """

    x = Cpt(EpicsMotor, ":mtr2", lazy=True)
    y = Cpt(EpicsMotor, ":mtr3", lazy=True)
    z = Cpt(EpicsMotor, ":mtr1", lazy=True)
