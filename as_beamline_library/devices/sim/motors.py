""" Simulated motor configuration and instantiation. """

from as_acquisition_library.devices.motors import ASSimMotor

from .classes.motors import MySimTable
from ophyd.sim import motor1, motor2, motor3

my_motor = ASSimMotor(name="my_motor")
my_table = MySimTable(name="my_table")

motor_x = motor1
motor_y = motor2
motor_z = motor3
