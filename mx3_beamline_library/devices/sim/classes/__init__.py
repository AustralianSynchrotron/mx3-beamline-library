"""This directory contains sim class definitions of devices specific to this beamline."""

from typing import Mapping


class Register:
    _registry = {}

    def __init__(self, name):
        self.name = name

    def __call__(self, cls):
        assert self.name not in self._registry
        self._registry[self.name] = cls

        return cls


def registry() -> Mapping[str, None]:
    """Returns a mapping from hardware database name to caproto class that handles it."""
    return Register._registry


from .detectors import BlackflyCamera  # noqa
