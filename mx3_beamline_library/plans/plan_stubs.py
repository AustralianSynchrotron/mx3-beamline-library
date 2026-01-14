import operator
import uuid
from functools import reduce
from time import sleep
from typing import Generator, Literal, Union

from bluesky.plan_stubs import create, mv, rd, read, save
from bluesky.tracing import trace_plan, tracer
from bluesky.utils import Msg, merge_cycler
from cycler import cycler
from ophyd import Device, Signal

from ..config import BL_ACTIVE
from ..devices.beam import transmission
from ..devices.classes.motors import MD3_CLIENT
from ..devices.motors import actual_sample_detector_distance, detector_fast_stage, md3
from ..logger import setup_logger

logger = setup_logger(__name__)

try:
    # cytools is a drop-in replacement for toolz, implemented in Cython
    from cytools import partition
except ImportError:
    from toolz import partition


@trace_plan(tracer, "md3_move")
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
        MD3_CLIENT.startSimultaneousMoveMotors(cmd)
        status = "running"
        while status == "running":
            sleep(0.02)
            status = MD3_CLIENT.getState().lower()
        yield Msg("wait", None, group=group)
    else:
        yield from mv(*args)


@trace_plan(tracer, "move_and_emit_document")
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


def get_fast_stage_setpoint(
    actual_detector_distance_setpoint: float,
) -> Generator[Msg, None, float]:
    """
    Calculates the fast stage setpoint given the actual_detector_distance_setpoint
    and the current actual_detector_distance_setpoint readback

    Parameters
    ----------
    actual_detector_distance_setpoint : float
        The sample detector distance


    Yields
    ------
    Generator[Msg, None, float]
        A bluesky message

    Raises
    ------
    ValueError
        If the setpoint is out of limits
    """
    actual_sample_detector_distance.wait_for_connection()
    actual_distance = yield from rd(actual_sample_detector_distance)
    diff = actual_detector_distance_setpoint - actual_distance

    detector_fast_stage.wait_for_connection()
    current_fast_stage_val = yield from rd(detector_fast_stage.user_readback)
    min_pos = yield from rd(detector_fast_stage.low_limit_travel)
    max_pos = yield from rd(detector_fast_stage.high_limit_travel)

    fast_stage_setpoint = current_fast_stage_val + diff

    limits = (min_pos, max_pos)

    if fast_stage_setpoint <= min_pos or fast_stage_setpoint >= max_pos:
        raise ValueError(
            f"Detector fast stage setpoint {fast_stage_setpoint} is out of limits: {limits}"
        )
    return fast_stage_setpoint


@trace_plan(tracer, "set_actual_sample_detector_distance")
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
    fast_stage_setpoint = yield from get_fast_stage_setpoint(
        actual_detector_distance_setpoint
    )
    yield from mv(detector_fast_stage, fast_stage_setpoint)


@trace_plan(tracer, "set_distance_and_transmission")
def set_distance_and_transmission(
    distance: float, transmission_value: float
) -> Generator[Msg, None, None]:
    """
    Sets the sample-detector distance and transmission asynchronously

    Parameters
    ----------
    distance : float
        The sample-detector distance in millimeters
    transmission_value : float
        The transmission value

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky message
    """
    if not 0 <= transmission_value <= 1:
        raise ValueError("Transmission must be a value between 0 and 1")

    fast_stage_setpoint = yield from get_fast_stage_setpoint(distance)
    yield from mv(
        detector_fast_stage, fast_stage_setpoint, transmission, transmission_value
    )


@trace_plan(tracer, "set_distance_and_md3_phase")
def set_distance_and_md3_phase(
    distance: float,
    md3_phase: Literal["Centring", "DataCollection", "BeamLocation", "Transfer"],
) -> Generator[Msg, None, None]:
    """
    Sets the sample-detector distance and md3 phase asynchronously

    Parameters
    ----------
    distance : float
        The sample-detector distance in millimeters
    md3_phase : Literal["Centring", "DataCollection", "BeamLocation", "Transfer"]
        The md3 phase

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky message
    """

    fast_stage_setpoint = yield from get_fast_stage_setpoint(distance)
    yield from mv(detector_fast_stage, fast_stage_setpoint, md3.phase, md3_phase)


@trace_plan(tracer, "set_distance_phase_and_transmission")
def set_distance_phase_and_transmission(
    distance: float,
    md3_phase: Literal["Centring", "DataCollection", "BeamLocation", "Transfer"],
    transmission_value: float,
):
    """
    Sets the sample-detector distance, md3 phase and transmission asynchronously

    Parameters
    ----------
    distance : float
        The sample-detector distance in millimeters
    md3_phase : Literal["Centring", "DataCollection", "BeamLocation", "Transfer"]
        The md3 phase
    transmission_value : float
        The transmission value
    Yields
    ------
    Generator[Msg, None, None]
        A bluesky message
    """
    if not 0 <= transmission_value <= 1:
        raise ValueError("Transmission must be a value between 0 and 1")
    fast_stage_setpoint = yield from get_fast_stage_setpoint(distance)
    yield from mv(
        detector_fast_stage,
        fast_stage_setpoint,
        md3.phase,
        md3_phase,
        transmission,
        transmission_value,
    )
