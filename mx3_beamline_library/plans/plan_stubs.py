import operator
import uuid
from functools import reduce
from typing import Generator, Union

from bluesky.plan_stubs import create, mv, rd, read, save
from bluesky.utils import Msg, merge_cycler
from cycler import cycler
from ophyd import Device, Signal

from ..config import BL_ACTIVE
from ..devices.classes.motors import SERVER
from ..devices.motors import actual_sample_detector_distance, detector_fast_stage
from ..logger import setup_logger

logger = setup_logger()

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


def set_actual_sample_detector_distance(
    actual_detector_distance_setpoint: float,
) -> Generator[Msg, None, None]:
    """
    Sets the actual detector distance from the sample to the detector
    by moving the detector fast stage

    Parameters
    ----------
    actual_detector_distance_setpoint : float
        The actual_detector_distance_setpoint in mm

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky message

    Raises
    ------
    ValueError
        Raises an error if the setpoint is out of the limits
        of the fast stage
    """
    actual_distance = actual_sample_detector_distance.get()
    if round(actual_distance, 2) == round(actual_detector_distance_setpoint, 2):
        logger.info(
            "Actual detector distance is already at setpoint: "
            f"{actual_detector_distance_setpoint}"
        )
        return
    diff = actual_detector_distance_setpoint - actual_distance
    current_fast_stage_val = yield from rd(detector_fast_stage)

    fast_stage_setpoint = current_fast_stage_val + diff

    limits = detector_fast_stage.limits
    if fast_stage_setpoint <= limits[0] or fast_stage_setpoint >= limits[1]:
        raise ValueError(
            f"Detector fast stage setpoint {fast_stage_setpoint} is out of limits: {limits}"
        )

    yield from mv(detector_fast_stage, fast_stage_setpoint)
