from datetime import datetime, timezone
from typing import Callable, Generator, Union

import numpy as np
from bluesky.plan_stubs import move_per_step, mv, read, trigger_and_read
from bluesky.plans import scan
from bluesky.utils import Msg
from ophyd import Signal
from ophyd.areadetector.base import EpicsSignalWithRBV
from ophyd.epics_motor import EpicsMotor

from ...devices.classes.detectors import GrasshopperCamera, HDF5Filewriter


def mx3_1d_scan(
    detectors: list[Union[GrasshopperCamera, HDF5Filewriter, EpicsSignalWithRBV]],
    motor: EpicsMotor,
    initial_position: float,
    final_position: float,
    number_of_steps: int,
    num: int = None,
    metadata: dict = None,
    hdf5_filename: str = None,
) -> Generator[Msg, None, None]:
    """
    Wrapper of the bluesky scan function for performing a 1D Gaussian distribution scan.

    This function runs a 1D scan until the scanned distribution is within one-sigma of the
    maximum value of the Gaussian.
    NOTE: The stop criteria for the scan is still under development and subject to change.

    Parameters
    ----------
    detectors : list[Union[GrasshopperCamera, HDF5Filewriter, EpicsSignalWithRBV]]
        A list of detectors, e.g., [my_camera.stats.total, my_camera.stats.sigma].
    motor : EpicsMotor
        The motor being scanned.
    initial_position : float
        The initial position of the motor.
    final_position : float
        The final position of the motor.
    number_of_steps : int
        The number of steps in the scan.
    num : int, optional
        Number of points, by default None
    metadata : dict, optional
        Additional metadata associated with the scan, by default None
    hdf5_filename : str, optional
        Name of the generated HDF5 file. If not provided, the filename is based
        on the generation time.
        Example: "mx3_1d_scan_28-08-2023_05:44:15.h5"

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan for the scan.

    Raises
    ------
    ValueError
        If metadata is not a dictionary.
    """

    if hdf5_filename is None:
        now = datetime.now(tz=timezone.utc)
        dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
        hdf5_filename = "mx3_1d_scan_" + dt_string + ".h5"

    if metadata is None:
        metadata = {"hdf5_filename": hdf5_filename}
    elif type(metadata) is dict:
        metadata.update({"hdf5_filename": hdf5_filename})
    elif type(metadata) is not dict:
        raise ValueError("Metadata must be a dictionary")

    for detector in detectors:
        detector.kind = "hinted"

        if type(detector) == HDF5Filewriter:
            detector: HDF5Filewriter
            yield from mv(
                detector.filename,
                hdf5_filename,
                detector.frames_per_datafile,
                number_of_steps,
            )
            write_path_template = detector.write_path_template.get()
            metadata.update({"write_path_template": write_path_template})

    # The following signals are used internally to save the total counts as a function of time,
    # and to determine if the plan should be stopped after certain criteria has been met
    _stats_buffer = Signal(name="buffer", kind="omitted", value=[])
    _stop_plan_signal = Signal(name="stop_plan", kind="omitted", value=False)
    detectors.append(_stop_plan_signal)
    detectors.append(
        _stats_buffer,
    )
    yield from scan(
        detectors,
        motor,
        initial_position,
        final_position,
        number_of_steps,
        num=num,
        per_step=_one_nd_step,
        md=metadata,
    )


def _one_nd_step(
    detectors: list,
    step: dict,
    pos_cache: dict,
    take_reading: Callable = trigger_and_read,
) -> Generator[Msg, None, None]:
    """
    Inner loop of the mx3_1d_scan

    Parameters
    ----------
    detectors : list
        devices to read
    step : dict
        mapping motors to positions in this step
    pos_cache : dict
        mapping motors to their last-set positions
    take_reading : Callable, optional
        function to do the actual acquisition ::

        Callable[List[OphydObj], Optional[str]] -> Generator[Msg], optional

        Defaults to `trigger_and_read`

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan for the scan.
    """
    _stop_plan_signal: Signal = detectors[-2]
    if not _stop_plan_signal.get():
        motors = step.keys()
        yield from move_per_step(step, pos_cache)
        reading = yield from take_reading(list(detectors) + list(motors))

        _stats_buffer = detectors[-1]
        for key in reading.keys():
            # TODO: for now we focus on stats.total, but in principle this parameter
            # can be anything we want
            key_index = key.find("total")
            if key_index >= 0:
                value = reading[key]["value"]
                stats_list = yield from read(_stats_buffer)
                stats_list["buffer"]["value"].append(value)
                yield from mv(_stats_buffer, stats_list["buffer"]["value"])

        stop_plan = _stop_plan(stats_list["buffer"]["value"])
        yield from mv(_stop_plan_signal, stop_plan)


def _stop_plan(stats_list: list) -> bool:
    """
    Criteria used to determine when to stop the mx3_1d_scan plan.

    This function provides the criteria for deciding when to stop the mx3_1d_scan plan based
    on the provided list of statistics.
    NOTE: The specific criteria for stopping the plan are still being refined and may
    change as development progresses.

    Parameters
    ----------
    stats_list : list
        A list of numerical values representing relevant statistics.

    Returns
    -------
    bool
        Returns True if the stop criteria have been met; otherwise, returns False.
    """

    if len(stats_list) < 2:
        return False
    else:
        second_derivative = np.gradient(np.gradient(stats_list))
        sign_diff = np.diff(np.sign(second_derivative))
        inflection_points = np.where(sign_diff)[0]
        if len(inflection_points) == 0:
            return False
        else:
            maximum_arg = np.where(second_derivative[inflection_points] < 0)[0]
            if len(maximum_arg) > 0:
                maximum_value = stats_list[inflection_points[maximum_arg][0]]
                difference = abs(maximum_value - stats_list[-1])
                # TODO: we now tolerate 3 sigma difference, but I just made up this number
                if difference > np.std(stats_list):
                    return True
                else:
                    return False
            else:

                return False
