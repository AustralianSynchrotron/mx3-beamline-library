from datetime import datetime, timezone
from typing import Callable, Generator, Union

from bluesky.plan_stubs import mv
from bluesky.plans import scan
from bluesky.utils import Msg
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
    per_step: Callable = None,
    metadata: dict = None,
    hdf5_filename: str = None,
) -> Generator[Msg, None, None]:
    """
    Wrapper of the bluesky scan function

    Parameters
    ----------
    detectors : list[Union[GrasshopperCamera, HDF5Filewriter, EpicsSignalWithRBV]]
        A list of detectors. Add here all parameters related to statistics that have to
        be calculated during the scan, e.g. [my_camera.stats.total, my_camera.stats.sigma]
    motor : EpicsMotor
        The motor used in the plan
    initial_position : float
        The motor initial position
    final_position : float
        the motor final position
    number_of_steps : int
        The number of steps of the scan
    num : int, optional
        Number of points, by default None
    per_step : Callable, optional
        Hook for customizing action of inner loop (messages per step).
        See docstring of bluesky.plan_stubs.one_nd_step() (the default)
        for details., by default None
    metadata : dict, optional
        Metadata, by default None
    hdf5_filename : str, optional
        The name of the HDF5 file generated during the run. If not provided,
        the file will be named based on the date and (UTC) time of generation,
        for example
        mx3_1d_scan_28-08-2023_05:44:15.h5

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan

    Raises
    ------
    ValueError
        An error if metadata is not a dictionary
    """
    if hdf5_filename is None:
        # datetime object containing current date and time
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
    yield from scan(
        detectors,
        motor,
        initial_position,
        final_position,
        number_of_steps,
        num=num,
        per_step=per_step,
        md=metadata,
    )
