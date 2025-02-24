"""
This code is provided AS IS for example purpose and testing MD Device Server
ARINAX Sep. 2021
"""

from .ExportClient import ExporterClientFactory


class ClientFactory:

    # @staticmethod
    def getTangoInstantiator(**kwargs):  # noqa
        # Do not import tango directly, as it is cumbersome to install
        from .TangoClient import TangoClientFactory

        return TangoClientFactory.instantiate(**kwargs)

    # @staticmethod
    def getEpicsInstantiator(**kwargs):  # noqa
        # Do not import epics directly, as it is cumbersome to install
        from .EpicsClient import EpicsClientFactory

        return EpicsClientFactory.instantiate(**kwargs)

    global impl
    impl = {
        "tango": getTangoInstantiator,
        "exporter": ExporterClientFactory.instantiate,
        "epics": getEpicsInstantiator,
    }

    @staticmethod
    def instantiate(*args, **kwargs):
        return impl[kwargs["type"]](**kwargs["args"])
