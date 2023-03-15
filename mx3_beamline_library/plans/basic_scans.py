""" """

import collections
import inspect
import logging
import time
from collections import defaultdict
from itertools import zip_longest
from os import environ
from time import sleep, perf_counter
from typing import Generator

import numpy as np
import numpy.typing as npt
from bluesky import plan_stubs as bps, preprocessors as bpp, utils
from bluesky.plan_stubs import configure, mv, stage, trigger_and_read, unstage  # noqa
from bluesky.plans import plan_patterns
from bluesky.utils import Msg
from ophyd import Device

from ..devices.classes.detectors import DectrisDetector
from ..devices.classes.md3.ClientFactory import ClientFactory
from ..devices.classes.motors import MD3Motor
from ..schemas.optical_and_xray_centering import (
    MD3ScanResponse,
    RasterGridMotorCoordinates,
)
from .stubs import stage_devices, unstage_devices

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)

MD3_ADDRESS = environ.get("MD3_ADDRESS", "10.244.101.30")
MD3_PORT = int(environ.get("MD3_PORT", 9001))

SERVER = ClientFactory.instantiate(
    type="exporter", args={"address": MD3_ADDRESS, "port": MD3_PORT}
)


def md3_grid_scan(
    detector: DectrisDetector,
    detector_configuration: dict,
    metadata: dict,
    grid_width: float,
    grid_height: float,
    start_omega: float,
    start_alignment_y: float,
    number_of_rows: int,
    start_alignment_z: float,
    start_sample_x: float,
    start_sample_y: float,
    number_of_columns: int,
    exposure_time: float,
    omega_range: float = 0,
    invert_direction: bool = True,
    use_centring_table: bool = True,
    use_fast_mesh_scans: bool = True,
) -> Generator[Msg, None, None]:
    """
    Bluesky plan that configures and arms the detector, the runs an md3 grid scan plan,
    and finally disarms the detector.

    Parameters
    ----------
    detector : DectrisDetector
        Dectris detector
    detector_configuration : dict
        Dictionary containing information about the configuration of the detector
    metadata : dict
        Plan metadata
    grid_width : float
        Width of the raster grid (mm)
    grid_height : float
        Height of the raster grid (mm)
    start_omega : float
        angle (deg) at which the shutter opens and omega speed is stable.
    number_of_rows : int
        Number of rows
    start_alignment_y : float
        Alignment y axis position at the beginning of the exposure
    start_alignment_z : float
        Alignment z axis position at the beginning of the exposure
    start_sample_x : float
        CentringX axis position at the beginning of the exposure
    start_sample_y : float
        CentringY axis position at the beginning of the exposure
    number_of_columns : int
        Number of columns
    exposure_time : float
        Exposure time measured in seconds to control shutter command. Note that
        this is the exposure time of one column only, e.g. the md3 takes
        `exposure_time` seconds to move `grid_height` mm.
    omega_range : float, optional
        Omega range (degrees) for the scan. This does not include the acceleration distance,
        by default 0
    invert_direction : bool, optional
        True to enable passes in the reverse direction, by default True
    use_centring_table : bool, optional
        True to use the centring table to do the pitch movements, by default True
    use_fast_mesh_scans : bool, optional
        True to use the fast raster scan if available (power PMAC), by default True

    Yields
    ------
    Generator
        A bluesky stub plan
    """
    metadata["dectris_sequence_id"] = detector.sequence_id.get()
    frame_rate = number_of_rows / exposure_time

    detector_configuration.update(
        {"trigger_mode": "exts", 
         "nimages": number_of_rows, 
         "frame_time": 1 / frame_rate, 
         "count_time": (1 / frame_rate) - 0.000001, # set count time explicitly
         "ntrigger": number_of_columns
        }
        )

    yield from configure(detector, detector_configuration)
    yield from stage(detector)

    # Rename variables to make the consistent with MD3 input parameters
    line_range = grid_height
    total_uturn_range = grid_width
    number_of_lines = number_of_columns
    frames_per_lines = number_of_rows

    t = perf_counter()
    raster_scan = SERVER.startRasterScanEx(
        omega_range,
        line_range,
        total_uturn_range,
        start_omega,
        start_alignment_y,
        start_alignment_z,
        start_sample_x,
        start_sample_y,
        number_of_lines,
        frames_per_lines,
        exposure_time,
        invert_direction,
        use_centring_table,
        use_fast_mesh_scans,
    )
    SERVER.waitAndCheck(
        task_name="Raster Scan",
        id=raster_scan,
        cmd_start=time.perf_counter(),
        expected_time=60,  # TODO: this should be estimated
        timeout=120,  # TODO: this should be estimated
    )
    logger.info(f"Execution time: {perf_counter() - t}")

    task_info = SERVER.retrieveTaskInfo(raster_scan)

    task_info_model = MD3ScanResponse(
        task_name=task_info[0],
        task_flags=task_info[1],
        start_time=task_info[2],
        end_time=task_info[3],
        task_output=task_info[4],
        task_exception=task_info[5],
        result_id=task_info[6],
    )
    print("Raster scan response:", raster_scan)
    logger.info(f"task info: {task_info_model.dict()}")
    
    yield from unstage(detector)

    return task_info_model  # noqa


def md3_4d_scan(
    detector: DectrisDetector,
    detector_configuration: dict,
    metadata: dict,
    start_angle: float,
    scan_range: float,
    exposure_time: float,
    start_alignment_y: float,
    start_alignment_z: float,
    start_sample_x: float,
    start_sample_y: float,
    stop_alignment_y: float,
    stop_alignment_z: float,
    stop_sample_x: float,
    stop_sample_y: float,
    number_of_frames: int,
) -> Generator[Msg, None, None]:
    """_summary_

    Parameters
    ----------
    detector : DectrisDetector
        A DectrisDetector ophy device
    detector_configuration : dict
        Detector configuration
    metadata : dict
        Metadata
    start_angle : float
        Start angle in degrees
    scan_range : float
        Scan range in degrees
    exposure_time : float
        Exposure time in seconds
    start_alignment_y : float
        Start alignment y
    start_alignment_z : float
        Start alignment z
    start_sample_x : float
        Start sample x 
    start_sample_y : float
        Start sample y
    stop_alignment_y : float
        Stop alignment y
    stop_alignment_z : float
        Stop alignment z 
    stop_sample_x : float
        Stop sample x
    stop_sample_y : float
        Stop sample y
    number_of_frames : int
        Number of frames, this parameter also corresponds to number of rows
        in a 1D grid scan

    Yields
    ------
    Generator
        A bluesky stub plan
    """
    metadata["dectris_sequence_id"] = detector.sequence_id.get()
    frame_rate = number_of_frames / exposure_time

    detector_configuration.update(
        {"trigger_mode": "exts", 
         "nimages": number_of_frames, 
         "frame_time": 1 / frame_rate, 
         "count_time": (1 / frame_rate) - 0.000001, # set count time explicitly
         "ntrigger": 1
        }
        )

    yield from configure(detector, detector_configuration)
    yield from stage(detector)

    scan_4d = SERVER.startScan4DEx(
        start_angle,
        scan_range,
        exposure_time,
        start_alignment_y,
        start_alignment_z,
        start_sample_x,
        start_sample_y,
        stop_alignment_y,
        stop_alignment_z,
        stop_sample_x,
        stop_sample_y,
    )
    SERVER.waitAndCheck(
        task_name="Raster Scan",
        id=scan_4d,
        cmd_start=time.perf_counter(),
        expected_time=60,  # TODO: this should be estimated
        timeout=120,  # TODO: this should be estimated
    )
    task_info = SERVER.retrieveTaskInfo(scan_4d)

    task_info_model = MD3ScanResponse(
        task_name=task_info[0],
        task_flags=task_info[1],
        start_time=task_info[2],
        end_time=task_info[3],
        task_output=task_info[4],
        task_exception=task_info[5],
        result_id=task_info[6],
    )

    print("Raster scan response:", scan_4d)
    logger.info(f"task info: {task_info_model.dict()}")
    yield from unstage(detector)

    return task_info_model  # noqa


def arm_trigger_and_disarm_detector(
    detector: Device, detector_configuration: dict, metadata: dict
) -> Generator[Msg, None, None]:
    """
    Bluesky plan that configures, arms, triggers and disarms the detector through
    the Simplon API

    Parameters
    ----------
    detector : Device
        Ophyd device
    detector_configuration : dict
        Dictionary containing information about the configuration of the detector

    Yields
    ------
    Generator
        A bluesky stub plan
    """
    yield from configure(detector, detector_configuration)
    yield from stage(detector)

    metadata["dectris_sequence_id"] = detector.sequence_id.get()

    # @bpp.run_decorator(md=metadata)
    # def inner():
    #    yield from trigger_and_read([detector])

    yield from trigger_and_read([detector])
    yield from unstage(detector)


def scan_nd(detectors, cycler, *, per_step=None, md=None):  # noqa
    """
    Scan over an arbitrary N-dimensional trajectory.

    Parameters
    ----------
    detectors : list
    cycler : Cycler
        cycler.Cycler object mapping movable interfaces to positions
    per_step : callable, optional
        hook for customizing action of inner loop (messages per step).
        See docstring of :func:`bluesky.plan_stubs.one_nd_step` (the default)
        for details.
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.inner_product_scan`
    :func:`bluesky.plans.grid_scan`

    Examples
    --------
    >>> from cycler import cycler
    >>> cy = cycler(motor1, [1, 2, 3]) * cycler(motor2, [4, 5, 6])
    >>> scan_nd([sensor], cy)
    """
    _md = {
        "detectors": [det.name for det in detectors],
        "motors": [motor.name for motor in cycler.keys],
        "num_points": len(cycler),
        "num_intervals": len(cycler) - 1,
        "plan_args": {
            "detectors": list(map(repr, detectors)),
            "cycler": repr(cycler),
            "per_step": repr(per_step),
        },
        "plan_name": "scan_nd",
        "hints": {},
    }
    _md.update(md or {})
    try:
        dimensions = [(motor.hints["fields"], "primary") for motor in cycler.keys]
    except (AttributeError, KeyError):
        # Not all motors provide a 'fields' hint, so we have to skip it.
        pass
    else:
        # We know that hints exists. Either:
        #  - the user passed it in and we are extending it
        #  - the user did not pass it in and we got the default {}
        # If the user supplied hints includes a dimension entry, do not
        # change it, else set it to the one generated above
        _md["hints"].setdefault("dimensions", dimensions)

    if per_step is None:
        per_step = bps.one_nd_step
    else:
        # Ensure that the user-defined per-step has the expected signature.
        sig = inspect.signature(per_step)

        def _verify_1d_step(sig):
            if len(sig.parameters) < 3:
                return False
            for name, (p_name, p) in zip_longest(
                ["detectors", "motor", "step"], sig.parameters.items()
            ):
                # this is one of the first 3 positional arguments,
                # check that the name matches
                if name is not None:
                    if name != p_name:
                        return False
                # if there are any extra arguments, check that they have a default
                else:
                    if p.kind is p.VAR_KEYWORD or p.kind is p.VAR_POSITIONAL:
                        continue
                    if p.default is p.empty:
                        return False

            return True

        def _verify_nd_step(sig):
            if len(sig.parameters) < 3:
                return False
            for name, (p_name, p) in zip_longest(
                ["detectors", "step", "pos_cache"], sig.parameters.items()
            ):
                # this is one of the first 3 positional arguments,
                # check that the name matches
                if name is not None:
                    if name != p_name:
                        return False
                # if there are any extra arguments, check that they have a default
                else:
                    if p.kind is p.VAR_KEYWORD or p.kind is p.VAR_POSITIONAL:
                        continue
                    if p.default is p.empty:
                        return False

            return True

        if sig == inspect.signature(bps.one_nd_step):
            pass
        elif _verify_nd_step(sig):
            # check other signature for back-compatibility
            pass
        elif _verify_1d_step(sig):
            # Accept this signature for back-compat reasons (because
            # inner_product_scan was renamed scan).
            dims = len(list(cycler.keys))
            if dims != 1:
                raise TypeError(
                    "Signature of per_step assumes 1D trajectory "
                    "but {} motors are specified.".format(dims)
                )
            (motor,) = cycler.keys
            user_per_step = per_step

            def adapter(detectors, step, pos_cache):
                # one_nd_step 'step' parameter is a dict; one_id_step 'step'
                # parameter is a value
                (step,) = step.values()
                return (yield from user_per_step(detectors, motor, step))

            per_step = adapter
        else:
            raise TypeError(
                "per_step must be a callable with the signature \n "
                "<Signature (detectors, step, pos_cache)> or "
                "<Signature (detectors, motor, step)>. \n"
                "per_step signature received: {}".format(sig)
            )
    pos_cache = defaultdict(lambda: None)  # where last position is stashed
    cycler = utils.merge_cycler(cycler)
    motors = list(cycler.keys)

    yield from stage_devices(list(detectors) + motors)

    _md["dectris_sequence_id"] = detectors[0].sequence_id.get()
    print("Sequence id: ", _md["dectris_sequence_id"])

    @bpp.run_decorator(md=_md)
    def inner_scan_nd():
        for step in list(cycler):
            yield from per_step(detectors, step, pos_cache)

    yield from inner_scan_nd()
    yield from unstage_devices(list(detectors) + motors)


def grid_scan(detectors, *args, snake_axes=None, per_step=None, md=None):  # noqa
    """
    Scan over a mesh; each motor is on an independent trajectory.

    Parameters
    ----------
    detectors: list
        list of 'readable' objects
    ``*args``
        patterned like (``motor1, start1, stop1, num1,``
                        ``motor2, start2, stop2, num2,``
                        ``motor3, start3, stop3, num3,`` ...
                        ``motorN, startN, stopN, numN``)

        The first motor is the "slowest", the outer loop. For all motors
        except the first motor, there is a "snake" argument: a boolean
        indicating whether to following snake-like, winding trajectory or a
        simple left-to-right trajectory.
    snake_axes: boolean or iterable, optional
        which axes should be snaked, either ``False`` (do not snake any axes),
        ``True`` (snake all axes) or a list of axes to snake. "Snaking" an axis
        is defined as following snake-like, winding trajectory instead of a
        simple left-to-right trajectory. The elements of the list are motors
        that are listed in `args`. The list must not contain the slowest
        (first) motor, since it can't be snaked.
    per_step: callable, optional
        hook for customizing action of inner loop (messages per step).
        See docstring of :func:`bluesky.plan_stubs.one_nd_step` (the default)
        for details.
    md: dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.rel_grid_scan`
    :func:`bluesky.plans.inner_product_scan`
    :func:`bluesky.plans.scan_nd`
    """
    # Notes: (not to be included in the documentation)
    #   The deprecated function call with no 'snake_axes' argument and 'args'
    #         patterned like (``motor1, start1, stop1, num1,``
    #                         ``motor2, start2, stop2, num2, snake2,``
    #                         ``motor3, start3, stop3, num3, snake3,`` ...
    #                         ``motorN, startN, stopN, numN, snakeN``)
    #         The first motor is the "slowest", the outer loop. For all motors
    #         except the first motor, there is a "snake" argument: a boolean
    #         indicating whether to following snake-like, winding trajectory or a
    #         simple left-to-right trajectory.
    #   Ideally, deprecated and new argument lists should not be mixed.
    #   The function will still accept `args` in the old format even if `snake_axes` is
    #   supplied, but if `snake_axes` is not `None` (the default value), it overrides
    #   any values of `snakeX` in `args`.

    args_pattern = plan_patterns.classify_outer_product_args_pattern(args)
    if (snake_axes is not None) and (
        args_pattern == plan_patterns.OuterProductArgsPattern.PATTERN_2
    ):
        raise ValueError(
            "Mixing of deprecated and new API interface is not allowed: "
            "the parameter 'snake_axes' can not be used if snaking is "
            "set as part of 'args'"
        )

    # For consistency, set 'snake_axes' to False if new API call is detected
    if (snake_axes is None) and (
        args_pattern != plan_patterns.OuterProductArgsPattern.PATTERN_2
    ):
        snake_axes = False

    chunk_args = list(plan_patterns.chunk_outer_product_args(args, args_pattern))
    # 'chunk_args' is a list of tuples of the form: (motor, start, stop, num, snake)
    # If the function is called using deprecated pattern for arguments, then
    # 'snake' may be set True for some motors, otherwise the 'snake' is always False.

    # The list of controlled motors
    motors = [_[0] for _ in chunk_args]

    # Check that the same motor is not listed multiple times.
    # This indicates an error in the script.
    if len(set(motors)) != len(motors):
        raise ValueError(
            f"Some motors are listed multiple times in"
            "the argument list 'args': "
            f"'{motors}'"
        )

    if snake_axes is not None:

        def _set_snaking(chunk, value):
            """Returns the tuple `chunk` with modified 'snake' value"""
            _motor, _start, _stop, _num, _snake = chunk
            return _motor, _start, _stop, _num, value

        if isinstance(snake_axes, collections.abc.Iterable) and not isinstance(
            snake_axes, str
        ):
            # Always convert to a tuple (in case a `snake_axes` is an iterator).
            snake_axes = tuple(snake_axes)

            # Check if the list of axes (motors) contains repeated entries.
            if len(set(snake_axes)) != len(snake_axes):
                raise ValueError(
                    f"The list of axes 'snake_axes'"
                    "contains repeated elements: "
                    f"'{snake_axes}'"
                )

            # Check if the snaking is enabled for the slowest motor.
            if len(motors) and (motors[0] in snake_axes):
                raise ValueError(
                    f"The list of axes 'snake_axes' "
                    "contains the slowest motor: "
                    f"'{snake_axes}'"
                )

            # Check that all motors in the chunk_args are controlled in the scan.
            #   It is very likely that the script running the plan has a bug.
            if any([_ not in motors for _ in snake_axes]):
                raise ValueError(
                    f"The list of axes 'snake_axes' contains motors "
                    f"that are not controlled during the scan: "
                    f"'{snake_axes}'"
                )

            # Enable snaking for the selected axes.
            #   If the argument `snake_axes` is specified (not None), then
            #   any `snakeX` values that could be specified in `args` are ignored.
            for n, chunk in enumerate(chunk_args):
                if n > 0:  # The slowest motor is never snaked
                    motor = chunk[0]
                    if motor in snake_axes:
                        chunk_args[n] = _set_snaking(chunk, True)
                    else:
                        chunk_args[n] = _set_snaking(chunk, False)

        elif snake_axes is True:  # 'snake_axes' has boolean value `True`
            # Set all 'snake' values except for the slowest motor
            chunk_args = [
                _set_snaking(_, True) if n > 0 else _ for n, _ in enumerate(chunk_args)
            ]
        elif snake_axes is False:  # 'snake_axes' has boolean value `True`
            # Set all 'snake' values
            chunk_args = [_set_snaking(_, False) for _ in chunk_args]
        else:
            raise ValueError(
                f"Parameter 'snake_axes' is not iterable,"
                "boolean or None: "
                f"'{snake_axes}', type: {type(snake_axes)}"
            )

    # Prepare the argument list for the `outer_product` function
    args_modified = []
    for n, chunk in enumerate(chunk_args):
        if n == 0:
            args_modified.extend(chunk[:-1])
        else:
            args_modified.extend(chunk)
    full_cycler = plan_patterns.outer_product(args=args_modified)

    md_args = []
    motor_names = []
    motors = []
    for i, (motor, start, stop, num, snake) in enumerate(chunk_args):
        md_args.extend([repr(motor), start, stop, num])
        if i > 0:
            # snake argument only shows up after the first motor
            md_args.append(snake)
        motor_names.append(motor.name)
        motors.append(motor)
    _md = {
        "shape": tuple(num for motor, start, stop, num, snake in chunk_args),
        "extents": tuple(
            [start, stop] for motor, start, stop, num, snake in chunk_args
        ),
        "snaking": tuple(snake for motor, start, stop, num, snake in chunk_args),
        # 'num_points': inserted by scan_nd
        "plan_args": {
            "detectors": list(map(repr, detectors)),
            "args": md_args,
            "per_step": repr(per_step),
        },
        "plan_name": "grid_scan",
        "plan_pattern": "outer_product",
        "plan_pattern_args": dict(args=md_args),
        "plan_pattern_module": plan_patterns.__name__,
        "motors": tuple(motor_names),
        "hints": {},
    }
    _md.update(md or {})
    _md["hints"].setdefault("gridding", "rectilinear")
    try:
        _md["hints"].setdefault(
            "dimensions", [(m.hints["fields"], "primary") for m in motors]
        )
    except (AttributeError, KeyError):
        ...

    return (yield from scan_nd(detectors, full_cycler, per_step=per_step, md=_md))


def _calculate_alignment_y_motor_coords(
    raster_grid_coords: RasterGridMotorCoordinates,
) -> npt.NDArray:
    """
    Calculates the y coordinates of the md3 grid scan

    Parameters
    ----------
    raster_grid_coords : RasterGridMotorCoordinates
        A RasterGridMotorCoordinates pydantic model

    Returns
    -------
    npt.NDArray
        The y coordinates of the md3 grid scan
    """

    if raster_grid_coords.number_of_rows == 1:
        # Especial case for number of rows == 1, otherwise
        # we get a division by zero
        motor_positions_array = np.array(
            [
                np.ones(raster_grid_coords.number_of_columns)
                * raster_grid_coords.initial_pos_alignment_y
            ]
        )

    else:
        delta_y = abs(
            raster_grid_coords.initial_pos_alignment_y
            - raster_grid_coords.final_pos_alignment_y
        ) / (raster_grid_coords.number_of_rows - 1)
        motor_positions_y = []
        for i in range(raster_grid_coords.number_of_rows):
            motor_positions_y.append(
                raster_grid_coords.initial_pos_alignment_y + delta_y * i
            )

        motor_positions_array = np.zeros(
            [raster_grid_coords.number_of_rows, raster_grid_coords.number_of_columns]
        )

        for i in range(raster_grid_coords.number_of_columns):
            if i % 2:
                motor_positions_array[:, i] = np.flip(motor_positions_y)
            else:
                motor_positions_array[:, i] = motor_positions_y

    return motor_positions_array


def _calculate_sample_x_coords(
    raster_grid_coords: RasterGridMotorCoordinates,
) -> npt.NDArray:
    """
    Calculates the sample_x coordinates of the md3 grid scan

    Parameters
    ----------
    raster_grid_coords : RasterGridMotorCoordinates
        A RasterGridMotorCoordinates pydantic model

    Returns
    -------
    npt.NDArray
        The sample_x coordinates of the md3 grid scan
    """
    if raster_grid_coords.number_of_columns == 1:
        motor_positions_array = np.array(
            [
                np.ones(raster_grid_coords.number_of_rows)
                * raster_grid_coords.center_pos_sample_x
            ]
        ).transpose()
    else:
        delta = abs(
            raster_grid_coords.initial_pos_sample_x
            - raster_grid_coords.final_pos_sample_x
        ) / (raster_grid_coords.number_of_columns - 1)

        motor_positions = []
        for i in range(raster_grid_coords.number_of_columns):
            motor_positions.append(raster_grid_coords.initial_pos_sample_x - delta * i)

        motor_positions_array = np.zeros(
            [raster_grid_coords.number_of_rows, raster_grid_coords.number_of_columns]
        )

        for i in range(raster_grid_coords.number_of_rows):
            motor_positions_array[i] = motor_positions

    return np.fliplr(motor_positions_array)


def _calculate_sample_y_coords(
    raster_grid_coords: RasterGridMotorCoordinates,
) -> npt.NDArray:
    """
    Calculates the sample_y coordinates of the md3 grid scan

    Parameters
    ----------
    raster_grid_coords : RasterGridMotorCoordinates
        A RasterGridMotorCoordinates pydantic model

    Returns
    -------
    npt.NDArray
        The sample_y coordinates of the md3 grid scan
    """
    if raster_grid_coords.number_of_columns == 1:
        motor_positions_array = np.array(
            [
                np.ones(raster_grid_coords.number_of_rows)
                * raster_grid_coords.center_pos_sample_y
            ]
        ).transpose()
    else:
        delta = abs(
            raster_grid_coords.initial_pos_sample_y
            - raster_grid_coords.final_pos_sample_y
        ) / (raster_grid_coords.number_of_columns - 1)

        motor_positions = []
        for i in range(raster_grid_coords.number_of_columns):
            motor_positions.append(raster_grid_coords.initial_pos_sample_y + delta * i)

        motor_positions_array = np.zeros(
            [raster_grid_coords.number_of_rows, raster_grid_coords.number_of_columns]
        )

        for i in range(raster_grid_coords.number_of_rows):
            motor_positions_array[i] = motor_positions

    return np.fliplr(motor_positions_array)


def test_md3_grid_scan_plan(
    raster_grid_coords: RasterGridMotorCoordinates,
    alignment_y: MD3Motor,
    sample_x: MD3Motor,
    sample_y: MD3Motor,
    omega: MD3Motor,
) -> Generator[Msg, None, None]:
    """
    This plan is used to reproduce an md3_grid_scan and validate the motor positions we use
    for the md3_grid_scan metadata. It is not intended to use in production.
    # TODO: Test this plan with more loops.

    Parameters
    ----------
    raster_grid_coords : RasterGridMotorCoordinates
        A RasterGridMotorCoordinates pydantic model
    alignment_y : MD3Motor
        Alignment_y
    sample_x : MD3Motor
        Sample_x
    sample_y: MD3Motor
        Sample_y

    Yields
    ------
    Generator[Msg, None, None]:
    """
    yield from mv(omega, raster_grid_coords.omega)

    y_array = _calculate_alignment_y_motor_coords(raster_grid_coords)
    sample_x_array = _calculate_sample_x_coords(raster_grid_coords)
    sample_y_array = _calculate_sample_y_coords(raster_grid_coords)
    for j in range(raster_grid_coords.number_of_columns):
        for i in range(raster_grid_coords.number_of_rows):
            yield from mv(
                alignment_y,
                y_array[i, j],
                sample_x,
                sample_x_array[i, j],
                sample_y,
                sample_y_array[i, j],
            )
            sleep(0.2)
