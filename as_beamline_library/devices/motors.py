""" Motor configuration and instantiation. """

from .classes.motors import Testrig
from ophyd import EpicsMotor

testrig = Testrig("MX3-testrig", name="testrig")

motor_x = testrig.x
# motor_x.wait_for_connection(timeout=5)
