""" Motor configuration and instantiation. """

from as_acquisition_library.devices.motors import ASEpicsMotor
from .classes.motors import MyTable

# my_motor = ASEpicsMotor("SR##ID01:MYMOTOR", name="my_motor")
# my_table = MyTable("SR##ID01:MYPREFIX", name="my_table")