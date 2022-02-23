""" Files in the top level of ``devices`` contain the instantiation of devices. The names of
    these files is at the discretion of Scientific Computing and Beamline Staff. However some 
    suggested files appear in the template. Do not prepend these file names with beamline 
    name/abbreviation.

    Imports are conditional on an environment variable ``BL_ACTIVE``. Sim devices are imported if not
    ``True``.
    
"""

from os import environ

try:
    if environ["BL_ACTIVE"] == "True":
        from . import motors
        from . import detectors
    else:
        from .sim import motors
        from .sim import detectors
except KeyError:
    from .sim import motors
    from .sim import detectors