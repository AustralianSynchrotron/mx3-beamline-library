import logging
import time
from os import environ
from time import perf_counter
from typing import Generator, Optional, Union

import numpy as np
import numpy.typing as npt
from bluesky.plan_stubs import configure, mv, stage, trigger, unstage
from bluesky.preprocessors import monitor_during_wrapper, run_wrapper
from bluesky.utils import Msg
from ophyd import Device, Signal

from ..devices.classes.detectors import DectrisDetector
from ..devices.classes.motors import SERVER, MD3Motor
from ..devices.detectors import dectris_detector
from ..devices.motors import md3
from ..schemas.crystal_finder import MotorCoordinates
from ..schemas.detector import DetectorConfiguration, UserData
from ..schemas.xray_centering import MD3ScanResponse, RasterGridCoordinates
from .plan_stubs import md3_move

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)

MD3_ADDRESS = environ.get("MD3_ADDRESS", "12.345.678.90")
MD3_PORT = int(environ.get("MD3_PORT", 1234))

MD3_SCAN_RESPONSE = Signal(name="md3_scan_response", kind="normal")
BL_ACTIVE = environ.get("BL_ACTIVE", "False").lower()


def _md3_scan(
    id: str,
    motor_positions: Union[MotorCoordinates, dict],
    number_of_frames: int,
    scan_range: float,
    exposure_time: float,
    number_of_passes: int = 1,
    tray_scan: bool = False,
    count_time: Optional[float] = None,
    drop_location: Optional[str] = None,
    hardware_trigger: bool = True,
) -> Generator[Msg, None, None]:
    """
    Runs an MD3 scan on a crystal.

    Parameters
    ----------
    id : str
        Id of the tray or sample
    motor_positions : Union[MotorCoordinates, dict]
        The motor positions at which the scan is done. The motor positions
        usually are inferred by the crystal finder. NOTE: We allow
        for dictionary types because the values sent via the
        bluesky queueserver do not support pydantic models
    number_of_frames : int
        The number of detector frames
    scan_range : float
        The range of the scan in degrees
    exposure_time : float
        The exposure time in seconds. NOTE: This is NOT the MD3 definition of exposure time
    number_of_passes : int, optional
        The number of passes, by default 1
    tray_scan : bool, optional
        Determines if the scan is done on a tray, by default False
    count_time : float, optional
        Detector count time. If this parameter is not set, it is set to
        frame_time - 0.0000001 by default. This calculation is done via
        the DetectorConfiguration pydantic model.
    drop_location: Optional[str]
        The location of the drop, used only when we screen trays, by default None
    hardware_trigger : bool, optional
        If set to true, we trigger the detector via hardware trigger, by default True.
        Warning! hardware_trigger=False is used mainly for development purposes,
        as it results in a very slow scan

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky stub plan
    """
    if type(motor_positions) is dict:
        motor_positions_model = MotorCoordinates(
            sample_x=motor_positions["sample_x"],
            sample_y=motor_positions["sample_y"],
            alignment_x=motor_positions["alignment_x"],
            alignment_y=motor_positions["alignment_y"],
            alignment_z=motor_positions["alignment_z"],
            omega=motor_positions["omega"],
            plate_translation=motor_positions.get("plate_translation"),
        )
    else:
        motor_positions_model = motor_positions

    if not tray_scan:
        yield from md3_move(
            md3.sample_x,
            motor_positions_model.sample_x,
            md3.sample_y,
            motor_positions_model.sample_y,
            md3.alignment_x,
            motor_positions_model.alignment_x,
            md3.alignment_y,
            motor_positions_model.alignment_y,
            md3.alignment_z,
            motor_positions_model.alignment_z,
            md3.omega,
            motor_positions_model.omega,
        )
    else:
        yield from md3_move(
            md3.sample_x,
            motor_positions_model.sample_x,
            md3.sample_y,
            motor_positions_model.sample_y,
            md3.alignment_x,
            motor_positions_model.alignment_x,
            md3.alignment_y,
            motor_positions_model.alignment_y,
            md3.alignment_z,
            motor_positions_model.alignment_z,
            md3.omega,
            motor_positions_model.omega,
            md3.plate_translation,
            motor_positions_model.plate_translation,
        )

    md3_exposure_time = number_of_frames * exposure_time

    frame_rate = number_of_frames / md3_exposure_time

    user_data = UserData(
        id=id, drop_location=drop_location, zmq_consumer_mode="filewriter"
    )

    detector_configuration = DetectorConfiguration(
        roi_mode="disabled",
        trigger_mode="exts",
        nimages=number_of_frames,
        frame_time=1 / frame_rate,
        count_time=count_time,
        ntrigger=number_of_passes,
        user_data=user_data,
    )

    yield from configure(
        dectris_detector, detector_configuration.dict(exclude_none=True)
    )

    yield from stage(dectris_detector)

    if BL_ACTIVE == "true":
        if hardware_trigger:
            scan_idx = 1  # NOTE: This does not seem to serve any useful purpose
            scan_id: int = SERVER.startScanEx2(
                scan_idx,
                number_of_frames,
                motor_positions_model.omega,
                scan_range,
                md3_exposure_time,
                number_of_passes,
            )
            cmd_start = time.perf_counter()
            timeout = 120
            SERVER.waitAndCheck(
                "Scan Omega", scan_idx, cmd_start, 3 + md3_exposure_time, timeout
            )
            task_info = SERVER.retrieveTaskInfo(scan_id)

            scan_response = MD3ScanResponse(
                task_name=task_info[0],
                task_flags=task_info[1],
                start_time=task_info[2],
                end_time=task_info[3],
                task_output=task_info[4],
                task_exception=task_info[5],
                result_id=task_info[6],
            )
            logger.info(f"task info: {scan_response.dict()}")

        else:
            scan_response = yield from _slow_scan(
                motor_positions=motor_positions_model,
                scan_range=scan_range,
                number_of_frames=number_of_frames,
            )

    elif BL_ACTIVE == "false":
        scan_response = yield from _slow_scan(
            motor_positions=motor_positions_model,
            scan_range=scan_range,
            number_of_frames=number_of_frames,
        )

    MD3_SCAN_RESPONSE.put(str(scan_response.dict()))
    yield from unstage(dectris_detector)
    yield from mv(md3.omega, 91.0)

    if scan_response.task_exception.lower() != "null":
        raise RuntimeError(f"Scan did not run successfully: {scan_response.dict()}")

    return scan_response


def md3_scan(
    id: str,
    motor_positions: Union[MotorCoordinates, dict],
    number_of_frames: int,
    scan_range: float,
    exposure_time: float,
    number_of_passes: int = 1,
    tray_scan: bool = False,
    count_time: Optional[float] = None,
    drop_location: Optional[str] = None,
    hardware_trigger: bool = True,
) -> Generator[Msg, None, None]:
    """
    Runs an MD3 scan on a crystal.

    Parameters
    ----------
    id : str
        Id of the tray or sample
    motor_positions : Union[MotorCoordinates, dict]
        The motor positions at which the scan is done. The motor positions
        usually are inferred by the crystal finder. NOTE: We allow
        for dictionary types because the values sent via the
        bluesky queueserver do not support pydantic models
    number_of_frames : int
        The number of detector frames
    scan_range : float
        The range of the scan in degrees
    exposure_time : float
        The exposure time in seconds
    number_of_passes : int, optional
        The number of passes, by default 1
    tray_scan : bool, optional
        Determines if the scan is done on a tray, by default False
    count_time : float, optional
        Detector count time. If this parameter is not set, it is set to
        frame_time - 0.0000001 by default. This calculation is done via
        the DetectorConfiguration pydantic model.
    drop_location: Optional[str]
        The location of the drop, used only when we screen trays, by default None
    hardware_trigger : bool, optional
        If set to true, we trigger the detector via hardware trigger, by default True.
        Warning! hardware_trigger=False is used mainly for development purposes,
        as it results in a very slow scan

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky stub plan
    """
    if drop_location is not None:
        metadata = {"id": id, "drop_location": drop_location}
    else:
        metadata = {"id": id}

    yield from monitor_during_wrapper(
        run_wrapper(
            _md3_scan(
                id=id,
                motor_positions=motor_positions,
                number_of_frames=number_of_frames,
                scan_range=scan_range,
                exposure_time=exposure_time,
                number_of_passes=number_of_passes,
                tray_scan=tray_scan,
                count_time=count_time,
                drop_location=drop_location,
                hardware_trigger=hardware_trigger,
            ),
            md=metadata,
        ),
        signals=([MD3_SCAN_RESPONSE]),
    )


def _slow_scan(
    motor_positions: MotorCoordinates, scan_range: float, number_of_frames: int
) -> Generator[Msg, None, None]:
    """
    Runs a scan on a crystal. This is a slow scan which triggers the detector
    via software trigger. Note: This plan is intended to be used only for
    development purposes.

    Parameters
    ----------
    motor_positions : MotorCoordinates
        The positions of the motors, usually obtained by the CrystalFinder
    scan_range : float
        The range of the scan in degrees
    number_of_frames : int
        The detector number of frames

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan
    """

    omega_array = np.linspace(
        motor_positions.omega, motor_positions.omega + scan_range, number_of_frames
    )
    for omega in omega_array:
        yield from mv(md3.omega, omega)
    # return a random response
    scan_response = MD3ScanResponse(
        task_name="Raster Scan",
        task_flags=8,
        start_time="2023-02-21 12:40:47.502",
        end_time="2023-02-21 12:40:52.814",
        task_output="org.embl.dev.pmac.PmacDiagnosticInfo@64ba4055",
        task_exception="null",
        result_id=1,
    )
    yield from trigger(dectris_detector)

    return scan_response


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
    md3_exposure_time: float,
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
    md3_exposure_time : float
        Exposure time measured in seconds to control shutter command. Note that
        this is the exposure time of one column, e.g. the md3 takes
        `md3_exposure_time` seconds to move `grid_height` mm.
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

    frame_rate = number_of_rows / md3_exposure_time

    detector_configuration = DetectorConfiguration(
        roi_mode="4M",
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
        md3_exposure_time,
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
    md3_exposure_time: float,
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
    md3_exposure_time : float
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
    frame_rate = number_of_frames / md3_exposure_time

    detector_configuration = DetectorConfiguration(
        roi_mode="4M",
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
        md3_exposure_time,
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

    yield from trigger(detector)
    yield from unstage(detector)


def _calculate_alignment_z_motor_coords(
    raster_grid_coords: RasterGridCoordinates,
) -> npt.NDArray:

    delta = abs(
        raster_grid_coords.initial_pos_alignment_z
        - raster_grid_coords.final_pos_alignment_z
    ) / (raster_grid_coords.number_of_columns - 1)

    motor_positions = []
    for i in range(raster_grid_coords.number_of_columns):
        motor_positions.append(
            raster_grid_coords.initial_pos_alignment_z + delta * i
        )  # check if this is plus or minus

    motor_positions_array = np.zeros(
        [raster_grid_coords.number_of_rows, raster_grid_coords.number_of_columns]
    )

    for i in range(raster_grid_coords.number_of_rows):
        motor_positions_array[i] = motor_positions

    return motor_positions_array


def _calculate_alignment_y_motor_coords(
    raster_grid_coords: RasterGridCoordinates,
) -> npt.NDArray:
    """
    Calculates the y coordinates of the md3 grid scan

    Parameters
    ----------
    raster_grid_coords : RasterGridCoordinates
        A RasterGridCoordinates pydantic model

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
    raster_grid_coords: RasterGridCoordinates,
) -> npt.NDArray:
    """
    Calculates the sample_x coordinates of the md3 grid scan

    Parameters
    ----------
    raster_grid_coords : RasterGridCoordinates
        A RasterGridCoordinates pydantic model

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
        sign = np.sign(np.sin(np.radians(raster_grid_coords.omega)))
        for i in range(raster_grid_coords.number_of_columns):
            motor_positions.append(
                raster_grid_coords.initial_pos_sample_x + sign * delta * i
            )

        motor_positions_array = np.zeros(
            [raster_grid_coords.number_of_rows, raster_grid_coords.number_of_columns]
        )

        for i in range(raster_grid_coords.number_of_rows):
            motor_positions_array[i] = motor_positions

    return np.fliplr(motor_positions_array)


def _calculate_sample_y_coords(
    raster_grid_coords: RasterGridCoordinates,
) -> npt.NDArray:
    """
    Calculates the sample_y coordinates of the md3 grid scan

    Parameters
    ----------
    raster_grid_coords : RasterGridCoordinates
        A RasterGridCoordinates pydantic model

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
        sign = np.sign(np.cos(np.radians(raster_grid_coords.omega)))
        for i in range(raster_grid_coords.number_of_columns):
            motor_positions.append(
                raster_grid_coords.initial_pos_sample_y + sign * delta * i
            )

        motor_positions_array = np.zeros(
            [raster_grid_coords.number_of_rows, raster_grid_coords.number_of_columns]
        )

        for i in range(raster_grid_coords.number_of_rows):
            motor_positions_array[i] = motor_positions

    return np.fliplr(motor_positions_array)


def slow_grid_scan(
    raster_grid_coords: RasterGridCoordinates,
    detector: DectrisDetector,
    detector_configuration: dict,
    alignment_y: MD3Motor,
    alignment_z: MD3Motor,
    sample_x: MD3Motor,
    sample_y: MD3Motor,
    omega: MD3Motor,
    use_centring_table: bool = True,
) -> Generator[Msg, None, None]:
    """
    This plan is used to reproduce an md3_grid_scan and validate the motor positions
    we use for the md3_grid_scan metadata. It is not intended to be used in
    production.

    Parameters
    ----------
    raster_grid_coords : RasterGridCoordinates
        A RasterGridCoordinates pydantic model
    detector : DectrisDetector
        The detector Ophyd device
    detector_configuration : dict
        The detector configuration
    alignment_y : MD3Motor
        Alignment_y
    alignment_z : MD3Motor
        Alignment_z
    sample_x : MD3Motor
        Sample_x
    sample_y: MD3Motor
        Sample_y
    omega: MD3Motor
        Omega
    use_centring_table : bool, optional
        If set to true we use the centring table, otherwise we
        run the grid scan using the alignment table, by default True

    Yields
    ------
    Generator[Msg, None, None]:
    """
    yield from mv(omega, raster_grid_coords.omega)

    yield from configure(detector, detector_configuration)
    yield from stage(detector)

    if use_centring_table:
        alignment_y_array = _calculate_alignment_y_motor_coords(raster_grid_coords)
        sample_x_array = _calculate_sample_x_coords(raster_grid_coords)
        sample_y_array = _calculate_sample_y_coords(raster_grid_coords)
        yield from mv(alignment_z, raster_grid_coords.initial_pos_alignment_z)

        for j in range(raster_grid_coords.number_of_columns):
            for i in range(raster_grid_coords.number_of_rows):
                yield from mv(alignment_y, alignment_y_array[i, j])
                yield from mv(sample_x, sample_x_array[i, j])
                yield from mv(sample_y, sample_y_array[i, j])
                yield from trigger(detector)

    else:
        alignment_y_array = _calculate_alignment_y_motor_coords(raster_grid_coords)
        alignment_z_array = _calculate_alignment_z_motor_coords(raster_grid_coords)

        for j in range(raster_grid_coords.number_of_columns):
            for i in range(raster_grid_coords.number_of_rows):
                yield from mv(alignment_y, alignment_y_array[i, j])
                yield from mv(alignment_z, alignment_z_array[i, j])
                yield from trigger(detector)

    yield from unstage(detector)

    # return a random response
    scan_response = MD3ScanResponse(
        task_name="Raster Scan",
        task_flags=8,
        start_time="2023-02-21 12:40:47.502",
        end_time="2023-02-21 12:40:52.814",
        task_output="org.embl.dev.pmac.PmacDiagnosticInfo@64ba4055",
        task_exception="null",
        result_id=1,
    )
    return scan_response
