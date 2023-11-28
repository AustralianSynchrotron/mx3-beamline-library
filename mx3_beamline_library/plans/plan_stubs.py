import operator
import uuid
from functools import reduce
from typing import Generator, Union

from bluesky.plan_stubs import create, mv, read, save
from bluesky.utils import Msg, merge_cycler
from cycler import cycler
from ophyd import Device, Signal

from ..config import BL_ACTIVE
from ..devices.classes.motors import SERVER

try:
    # cytools is a drop-in replacement for toolz, implemented in Cython
    from cytools import partition
except ImportError:
    from toolz import partition


def md3_move(*args, group: str = None) -> Generator[Msg, None, None]:
    """
    Move one or more md3 motors to a setpoint. Wait for all to complete.
    If more than one device is specified, the movements are done in parallel.

    Parameters
    ----------
    args :
        device1, value1, device2, value2, ...
    group : str, optional
        Used to mark these as a unit to be waited on.


    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan

    """
    group = group or str(uuid.uuid4())

    cyl = reduce(operator.add, [cycler(obj, [val]) for obj, val in partition(2, args)])
    (step,) = merge_cycler(cyl)

    cmd = str()
    for obj, val in step.items():
        cmd += f"{obj.name}={val},"

    if BL_ACTIVE == "true":
        SERVER.startSimultaneousMoveMotors(cmd)
        status = "running"
        while status == "running":
            status = SERVER.getState().lower()
        yield Msg("wait", None, group=group)
    else:
        yield from mv(*args)


def move_and_emit_document(
    signal: Union[Signal, Device], value: Union[str, float, dict]
) -> Generator[Msg, None, None]:
    """
    Moves a signal and emits a document

    Parameters
    ----------
    signal : Union[Signal, Device]
        A signal or device
    value : Union[str, float, dict]
        The new value

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan
    """
    yield from create(name=signal.name)
    yield from mv(signal, value)
    yield from read(signal)
    yield from save()
