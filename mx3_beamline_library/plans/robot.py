from time import sleep
from typing import Generator, Union

from bluesky.plan_stubs import close_run, mv, open_run
from bluesky.utils import Msg
from mx_robot_library.schemas.common.sample import Pin

from ..devices.motors import isara_robot, md3
from .plan_stubs import set_actual_sample_detector_distance, set_distance_and_md3_phase


def mount_pin(
    pin: Pin, prepick_pin: Union[Pin, None] = None
) -> Generator[Msg, None, None]:
    """
    Mounts a pin given an id and puck, and then changes the phase of the MD3
    to `Centring` mode.

    Parameters
    ----------
    pin : Pin
        The pin to mount
    prepick_pin : Union[Pin, None], optional
        Prepick pin, by default None

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan
    """
    yield from open_run()
    sample_detector_distance = 380  # mm

    if md3.phase.get() != "Transfer":
        yield from set_distance_and_md3_phase(sample_detector_distance, "Transfer")
    else:
        yield from set_actual_sample_detector_distance(sample_detector_distance)

    yield from mv(isara_robot.mount, {"pin": pin, "prepick_pin": prepick_pin})
    yield from mv(md3.phase, "Centring")
    yield from close_run()


def unmount_pin() -> Generator[Msg, None, None]:
    """
    Changes the phase of the md3 to `Transfer` mode, and then unmounts a pin.

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky stub plan
    """
    yield from open_run()
    sample_detector_distance = 380  # mm

    if md3.phase.get() != "Transfer":
        yield from set_distance_and_md3_phase(sample_detector_distance, "Transfer")
    else:
        yield from set_actual_sample_detector_distance(sample_detector_distance)

    yield from mv(isara_robot.unmount, None)
    yield from close_run()


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
    sample_detector_distance = 380  # mm

    if md3.phase.get() != "Transfer":
        yield from set_distance_and_md3_phase(sample_detector_distance, "Transfer")
    else:
        yield from set_actual_sample_detector_distance(sample_detector_distance)

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

    sample_detector_distance = 380  # mm

    if md3.phase.get() != "Transfer":
        yield from set_distance_and_md3_phase(sample_detector_distance, "Transfer")
    else:
        yield from set_actual_sample_detector_distance(sample_detector_distance)

    yield from mv(isara_robot.unmount_tray, None)


def vegas_mode(
    puck: int = 1,
    max_id: int = 16,
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
        Maximum id number of a puck, by default 16

    Yields
    ------
    Generator[Msg, None, None]
        Bluesky messages
    """

    while True:
        for i in range(1, max_id):
            yield from mv(md3.phase, "Transfer")
            yield from mount_pin(
                pin=Pin(id=i, puck=puck), prepick_pin=Pin(id=(i + 1), puck=puck)
            )
            sleep(1)
            yield from mv(md3.phase, "Transfer")
            if i == max_id - 1:
                yield from unmount_pin()
            sleep(1)
            yield from mv(md3.phase, "Transfer")
