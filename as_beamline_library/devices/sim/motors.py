"""
Simulated Motor Definitions
"""

from as_acquisition_library.devices.motors import ASSimMotor
from .classes.motors import MySimTable

my_motor = ASSimMotor(name="my_motor")
my_table = MySimTable(name="my_table")
