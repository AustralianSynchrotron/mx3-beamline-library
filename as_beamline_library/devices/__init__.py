""" Files in the top level of *_beamline_library contain the instantiation of devices. The names of these files is at the discretion of Scientific Computing and Beamline Staff"""

from os import environ

try:
    if environ["BL_ACTIVE"] == "True":
        from . import motors
    else:
        from .sim import motors
except KeyError:
    from .sim import motors
