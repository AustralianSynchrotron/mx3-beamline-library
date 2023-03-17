""" """

import logging
import time
from os import environ
from time import perf_counter, sleep
from typing import Generator, Optional

import numpy as np
import numpy.typing as npt
from bluesky.plan_stubs import configure, mv, stage, trigger_and_read, unstage  # noqa
from bluesky.utils import Msg
from ophyd import Device

from ..devices.classes.detectors import DectrisDetector
from ..devices.classes.md3.ClientFactory import ClientFactory
from ..devices.classes.motors import MD3Motor
from ..schemas.detector import DetectorConfiguration, UserData
from ..schemas.optical_and_xray_centering import (
    MD3ScanResponse,
    RasterGridMotorCoordinates,
)

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)

MD3_ADDRESS = environ.get("MD3_ADDRESS", "12.345.678.90")
MD3_PORT = int(environ.get("MD3_PORT", 1234))

SERVER = ClientFactory.instantiate(
    type="exporter", args={"address": MD3_ADDRESS, "port": MD3_PORT}
)


def md3_grid_scan(
    detector: DectrisDetector,
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
    user_data: Optional[UserData] = None,
    count_time: Optional[float] = None,
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
    user_data : UserData, optional
        User data pydantic model. This field is passed to the start message
        of the ZMQ stream
    count_time : float, optional
        Detector count time. If this parameter is not set, it is set to
        frame_time - 0.0000001 by default. This calculation is done via
        the DetectorConfiguration pydantic model.

    Yields
    ------
    Generator
        A bluesky stub plan
    """
    assert number_of_columns > 1, "Number of columns must be > 1"

    frame_rate = number_of_rows / exposure_time

    detector_configuration = DetectorConfiguration(
        trigger_mode="exts",
        nimages=number_of_rows,
        frame_time=1 / frame_rate,
        count_time=count_time,
        ntrigger=number_of_columns,
        user_data=user_data,
    )

    yield from configure(detector, detector_configuration.dict(exclude_none=True))

    yield from stage(detector)

    # Rename variables to make them consistent with MD3 input parameters
    line_range = grid_height
    total_uturn_range = grid_width
    number_of_lines = number_of_columns
    frames_per_lines = number_of_rows

    t = perf_counter()
    # NOTE: The scan_id is stored in the MD3ScanResponse,
    # and is also sent via bluesky documents
    scan_id: int = SERVER.startRasterScanEx(
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
        id=scan_id,
        cmd_start=time.perf_counter(),
        expected_time=60,  # TODO: this should be estimated
        timeout=120,  # TODO: this should be estimated
    )
    logger.info(f"Execution time: {perf_counter() - t}")

    task_info = SERVER.retrieveTaskInfo(scan_id)

    task_info_model = MD3ScanResponse(
        task_name=task_info[0],
        task_flags=task_info[1],
        start_time=task_info[2],
        end_time=task_info[3],
        task_output=task_info[4],
        task_exception=task_info[5],
        result_id=task_info[6],
    )
    logger.info(f"task info: {task_info_model.dict()}")

    yield from unstage(detector)

    return task_info_model  # noqa


def md3_4d_scan(
    detector: DectrisDetector,
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
    user_data: Optional[UserData] = None,
    count_time: Optional[float] = None,
) -> Generator[Msg, None, None]:
    """
    Runs an md3 4d scan. This plan is also used for running a 1D grid scan, since setting
    number_of_columns=1 on the md3_grid_scan raises an issue

    Parameters
    ----------
    detector : DectrisDetector
        A DectrisDetector ophyd device
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
    user_data : UserData, optional
        User data pydantic model. This field is passed to the start message
        of the ZMQ stream
    count_time : float, optional
        Detector count time. If this parameter is not set, it is set to
        frame_time - 0.0000001 by default. This calculation is done via
        the DetectorConfiguration pydantic model.

    Yields
    ------
    Generator
        A bluesky stub plan
    """
    frame_rate = number_of_frames / exposure_time

    detector_configuration = DetectorConfiguration(
        trigger_mode="exts",
        nimages=number_of_frames,
        frame_time=1 / frame_rate,
        count_time=count_time,
        ntrigger=1,
        user_data=user_data,
    )

    yield from configure(detector, detector_configuration.dict(exclude_none=True))
    yield from stage(detector)

    # NOTE: The scan_id is stored in the MD3ScanResponse,
    # and is also sent via bluesky documents
    scan_id: int = SERVER.startScan4DEx(
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
        id=scan_id,
        cmd_start=time.perf_counter(),
        expected_time=60,  # TODO: this should be estimated
        timeout=120,  # TODO: this should be estimated
    )
    task_info = SERVER.retrieveTaskInfo(scan_id)

    task_info_model = MD3ScanResponse(
        task_name=task_info[0],
        task_flags=task_info[1],
        start_time=task_info[2],
        end_time=task_info[3],
        task_output=task_info[4],
        task_exception=task_info[5],
        result_id=task_info[6],
    )

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
