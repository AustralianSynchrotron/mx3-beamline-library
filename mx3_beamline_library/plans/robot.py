from typing import Generator

from bluesky.plan_stubs import mv
from bluesky.utils import Msg

from ..devices.motors import isara_robot, md3


def mount_pin(id: int, puck: int) -> Generator[Msg, None, None]:
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
    yield from mv(isara_robot.mount, {"id": id, "puck": puck})
    yield from mv(md3.phase, "Centring")


def unmount_pin() -> Generator[Msg, None, None]:
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
    yield from mv(md3.phase, "Transfer")
    yield from mv(isara_robot.unmount, None)


def mount_tray(id: int) -> Generator[Msg, None, None]:
    """
    Mounts a tray on the MD3.  Note that at the moment we set the alignment_y value
    to -7.5 to solve the MD3 Arinax bug

    Parameters
    ----------
    id : int
        ID of the plate

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan
    """
    yield from mv(md3.phase, "Transfer")
    # NOTE: this solves the ARINAX bug momentarily
    yield from mv(md3.alignment_y, -7.5)
    yield from mv(isara_robot.mount_tray, id)


def unmount_tray() -> Generator[Msg, None, None]:
    """
    Unmounts a tray. Note that at the moment we set the alignment_y value
    to -7.5 to solve the MD3 Arinax bug

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan
    """

    yield from mv(md3.phase, "Transfer")
    # NOTE: this solves the ARINAX bug momentarily
    yield from mv(md3.alignment_y, -7.5)
    yield from mv(isara_robot.unmount_tray, None)


def vegas_mode(
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
            yield from mv(md3.phase, "Transfer")
            yield from mount_pin(id=i, puck=puck)
            yield from mv(md3.phase, "Transfer")
            yield from unmount_pin()
            yield from mv(md3.phase, "Transfer")
