""" This directory contains class definitions for devices specific to this beamline. """
from typing import Mapping

from ophyd import Device



class Register:
    _registry = {}

    def __init__(self, name):
        self.name = name

    def __call__(self, cls):
        assert self.name not in self._registry
        self._registry[self.name] = cls

        return cls

def registry() -> Mapping[str,Device]:
    """Returns a mapping from hardware database name to ophyd class that handles it."""
    return Register._registry

#from .camera import MantaCamera
#from .dtacq import DTACQ
