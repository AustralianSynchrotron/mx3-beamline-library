from ..devices.classes.robot import Mount, Unmount
from typing import Generator
from bluesky.utils import Msg
from bluesky.plan_stubs import mv

def mount_pin(mount_signal: Mount, id: int, puck: int) -> Generator[Msg, None, None]:
    """
    Mounts a pin given an id and puck

    Parameters
    ----------
    mount_signal : Mount
        A robot mount signal
    id : int
        id
    puck : int
        Puck

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky stub plan
    """
    yield from mv(mount_signal, {"id": id ,"puck": puck})

def unmount_pin(unmount_signal: Unmount) -> Generator[Msg, None, None]:
    """
    Unmounts a pin

    Parameters
    ----------
    unmount_signal : Unmount
       A robot unmount signal

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky stub plan
    """
    yield from mv(unmount_signal, None)
