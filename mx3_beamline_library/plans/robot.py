from typing import Generator

from bluesky.plan_stubs import mv
from bluesky.utils import Msg

from ..devices.classes.motors import MD3Phase
from ..devices.classes.robot import Mount, Unmount


def mount_pin(mount_signal: Mount, md3_phase_signal: MD3Phase, id: int, puck: int) -> Generator[Msg, None, None]:
    """
    Mounts a pin given an id and puck, and then changes the phase of the MD3
    to `Centring` mode.

    Parameters
    ----------
    mount_signal : Mount
        A robot mount signal
    md3_phase_signal : MD3Phase
        md3_phase_signal
    id : int
        id
    puck : int
        Puck

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky stub plan
    """
    yield from mv(mount_signal, {"id": id, "puck": puck})
    yield from mv(md3_phase_signal, "Centring")


def unmount_pin(unmount_signal: Unmount, md3_phase_signal: MD3Phase) -> Generator[Msg, None, None]:
    """
    Changes the phase of the md3 to `Transfer` mode, and then unmounts a pin.

    Parameters
    ----------
    unmount_signal : Unmount
       A robot unmount signal
    md3_phase_signal : MD3Phase
        MD3 Phase signal

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky stub plan
    """
    yield from mv(md3_phase_signal, "Transfer")
    yield from mv(unmount_signal, None)


def vegas_mode(
    phase_signal: MD3Phase,
    mount_signal: Mount,
    unmount_signal: Unmount,
    puck: int = 1,
    max_id: int = 17,
) -> Generator[Msg, None, None]:
    """
    Unmounts and mounts samples in a loop.
    Warning! This plan will run until it's manually stopped, and is
    intended for demonstration purposes only.

    Parameters
    ----------
    phase_signal : MD3Phase
        Md3 Phase
    mount_signal : Mount
        Mount Signal
    unmount_signal : Unmount
        Unmount signal
    puck : int, Optional
        Puck id, be default 1
    max_id : int, Optional
        Maximum id number of a puck, by default 17

    Yields
    ------
    Generator[Msg, None, None]
        Bluesky messages
    """

    while True:
        for i in range(1, max_id):
            yield from mv(phase_signal, "Transfer")
            yield from mount_pin(mount_signal, id=i, puck=puck)
            yield from mv(phase_signal, "Transfer")
            yield from unmount_pin(unmount_signal)
            yield from mv(phase_signal, "Transfer")
