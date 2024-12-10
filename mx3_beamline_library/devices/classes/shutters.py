from enum import IntEnum, unique
from typing import Any

from ophyd import Component, EpicsSignal, EpicsSignalRO
from ophyd.pv_positioner import PVPositionerComparator


@unique
class OpenCloseCmd(IntEnum):
    NO_ACTION = 0
    CLOSE_VALVE = 1
    OPEN_VALVE = 2


@unique
class OpenCloseStatus(IntEnum):
    UNKNOWN = 0
    INVALID = 1  # MAJOR
    CLOSED = 2  # NO_ALARM
    OPEN = 3  # NO_ALARM
    MOVING = 4  # NO_ALARM


class SinglePSSShutters(PVPositionerComparator):
    open_close_cmd = Component(EpicsSignal, ":OPEN_CLOSE_CMD")
    """Control to shutter, Enum
     Write: OpenCloseCmd
     """

    open_close_status = Component(EpicsSignalRO, ":OPEN_CLOSE_STATUS")
    """Shutter status, Enum
    Read: OpenCloseStatus
    """

    open_enabled = Component(EpicsSignalRO, ":OPEN_ENABLED_STATUS")
    """Bool
    Read:
    1 Shutter is open enabled
    0 Disabled (NO_ALARM)
    """

    setpoint = open_close_cmd
    readback = open_close_status

    def __init__(self, prefix: str, *, name: str, **kwargs):
        kwargs.update({"limits": (OpenCloseCmd.CLOSE_VALVE, OpenCloseCmd.OPEN_VALVE)})
        super().__init__(prefix, name=name, **kwargs)

    def done_comparator(self, readback: Any, setpoint: Any) -> bool:
        return readback != OpenCloseStatus.MOVING
