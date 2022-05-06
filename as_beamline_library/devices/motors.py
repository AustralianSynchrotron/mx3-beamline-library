""" Motor configuration and instantiation. """

from .classes.motors import Testrig

testrig = Testrig("MX3-testrig", name="testrig")

motor_x = testrig.x
motor_y = testrig.y
motor_z = testrig.z
