import operator
import uuid
from functools import reduce
from os import environ
from typing import Generator

from bluesky.plan_stubs import mv
from bluesky.utils import Msg, merge_cycler
from cycler import cycler

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

    if environ["BL_ACTIVE"].lower() == "true":
        SERVER.startSimultaneousMoveMotors(cmd)
        status = "running"
        while status == "running":
            status = SERVER.getState().lower()
        yield Msg("wait", None, group=group)
    else:
        yield from mv(*args)
