# from ..devices.classes.motors import SERVER, MD3Motor
import operator
import uuid
from functools import reduce
from os import environ
from time import sleep

from bluesky.plan_stubs import mv
from bluesky.utils import Msg, merge_cycler
from cycler import cycler

from ..devices.classes.motors import SERVER

try:
    # cytools is a drop-in replacement for toolz, implemented in Cython
    from cytools import partition
except ImportError:
    from toolz import partition


def md3_move(*args, group: str = None):
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
    msg : Msg

    """
    group = group or str(uuid.uuid4())

    cyl = reduce(operator.add, [cycler(obj, [val]) for obj, val in partition(2, args)])
    (step,) = merge_cycler(cyl)

    cmd = str()
    for obj, val in step.items():
        cmd += f"{obj.name}={val},"

    if environ["BL_ACTIVE"].lower() == "true":
        SERVER.startSimultaneousMoveMotors(cmd)
        status = "Running"
        while status == "Running":
            status = SERVER.getState()
            sleep(0.2)
        yield Msg("wait", None, group=group)
    else:
        yield from mv(*args)
