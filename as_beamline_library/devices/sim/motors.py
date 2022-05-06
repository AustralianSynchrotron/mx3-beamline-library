""" Simulated motor configuration and instantiation. """

from as_acquisition_library.devices.motors import ASSimMotor

from .classes.motors import MySimTable

my_motor = ASSimMotor(name="my_motor")

# Simulated testrig motors
testrig = MySimTable(name="testrig")
