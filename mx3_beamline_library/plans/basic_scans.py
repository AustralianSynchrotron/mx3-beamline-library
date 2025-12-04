import time
from time import perf_counter
from typing import Generator, Literal, Optional
from uuid import UUID

import numpy as np
import numpy.typing as npt
from bluesky.plan_stubs import configure, mv, stage, trigger, unstage
from bluesky.preprocessors import monitor_during_wrapper, run_wrapper
from bluesky.tracing import trace_plan, tracer
from bluesky.utils import Msg
from ophyd import Signal

from ..config import BL_ACTIVE
from ..devices.classes.detectors import DectrisDetector
from ..devices.classes.motors import MD3_CLIENT, MD3Motor
from ..devices.detectors import dectris_detector
from ..devices.motors import md3
from ..logger import setup_logger
from ..schemas.crystal_finder import MotorCoordinates
from ..schemas.detector import DetectorConfiguration, UserData
from ..schemas.xray_centering import MD3ScanResponse, RasterGridCoordinates
from .beam_utils import set_beam_center
from .crystal_pics import save_crystal_pic_to_redis
from .exceptions import MD3Exception
from .plan_stubs import md3_move, set_distance_phase_and_transmission

logger = setup_logger(__name__)


MD3_SCAN_RESPONSE = Signal(name="md3_scan_response", kind="normal")


def _start_md3_scan(
    number_of_frames: int,
    omega_start: float,
    scan_range: float,
    md3_exposure_time: float,
    number_of_passes: int,
    tray_scan: bool,
    hardware_trigger: bool,
) -> Generator[Msg, None, MD3ScanResponse]:
    """
    Helper function that starts an MD3 scan in sim or real device mode.

    Parameters
    ----------
    number_of_frames : int
        The number of detector frames
    omega_start : float
        The initial omega angle
    scan_range : float
        The range of the scan in degrees
    md3_exposure_time : float
        The MD3 exposure time in seconds
    number_of_passes : int
        The number of passes
    tray_scan : bool
        Whether the scan is a tray scan
    hardware_trigger : bool
        Whether to use hardware trigger

    Yields
    ------
    Generator[Msg, None, MD3ScanResponse]
        A bluesky plan containing the MD3 scan response
    """
    if BL_ACTIVE == "true" and hardware_trigger:
        scan_idx = 1
        scan_id: int = MD3_CLIENT.startScanEx2(
            scan_idx,
            number_of_frames,
            omega_start,
            scan_range,
            md3_exposure_time,
            number_of_passes,
        )
        cmd_start = time.perf_counter()
        timeout = 120
        MD3_CLIENT.waitAndCheck(
            "Scan Omega", scan_idx, cmd_start, 3 + md3_exposure_time, timeout
        )
        task_info = MD3_CLIENT.retrieveTaskInfo(scan_id)

        scan_response = MD3ScanResponse(
            task_name=task_info[0],
            task_flags=task_info[1],
            start_time=task_info[2],
            end_time=task_info[3],
            task_output=task_info[4],
            task_exception=task_info[5],
            result_id=task_info[6],
        )
        logger.info(f"task info: {scan_response.model_dump()}")
        if scan_response.task_exception.lower() != "null":
            raise MD3Exception(
                f"The Scan failed with error {scan_response.task_exception}. "
                f"Response: {scan_response.model_dump()}"
            )

    else:
        scan_response = yield from _slow_scan(
            start_omega=md3.omega.position,
            scan_range=scan_range,
            number_of_frames=number_of_frames,
            tray_scan=tray_scan,
        )
    return scan_response


def determine_start_omega(
    motor_positions: MotorCoordinates | None, scan_range: float, tray_scan: bool
) -> float:
    """
    Helper function to determine the start omega position for an MD3 scan.
    If tray_scan=True, the start angle is either a) 91 - scan_range/2
    or b) 270 - scan_range/2. If motor positions is None and tray_scan is False,
    the current omega position of the md3 motor is used.

    Parameters
    ----------
    motor_positions : MotorCoordinates | None
        The motor positions at which the scan is done. The motor positions
        usually are inferred by the crystal finder
    scan_range : float
        The range of the scan in degrees
    tray_scan : bool
        Whether the scan is a tray scan
    """

    if tray_scan:
        if scan_range > 30:
            raise ValueError(
                "Scan range for trays cannot exceed 30 degrees. Decrease the scan range"
            )
        omega_position = md3.omega.position
        if 70 <= omega_position <= 110:
            return 91 - scan_range / 2
        if 250 <= omega_position <= 290:
            return 270 - scan_range / 2
        raise ValueError(
            "Start omega should either be in the range (70,110) "
            f"or (250,290). Current value is {omega_position}"
        )

    if motor_positions is not None:
        return motor_positions.omega

    return md3.omega.position


def _move_to_motor_coordinates(
    motor_positions: MotorCoordinates | None, tray_scan: bool
) -> Generator[Msg, None, float]:
    """
    Helper function to move to the motor coordinates for an MD3 scan.

    Parameters
    ----------
    motor_positions : MotorCoordinates | None
        The motor positions at which the scan is done. The motor positions
        usually are inferred by the crystal finder
    tray_scan : bool
        Whether the scan is a tray scan
    """

    if tray_scan:
        yield from md3_move(
            md3.sample_x,
            motor_positions.sample_x,
            md3.sample_y,
            motor_positions.sample_y,
            md3.alignment_x,
            motor_positions.alignment_x,
            md3.alignment_y,
            motor_positions.alignment_y,
            md3.alignment_z,
            motor_positions.alignment_z,
            md3.plate_translation,
            motor_positions.plate_translation,
        )

    else:
        yield from md3_move(
            md3.sample_x,
            motor_positions.sample_x,
            md3.sample_y,
            motor_positions.sample_y,
            md3.alignment_x,
            motor_positions.alignment_x,
            md3.alignment_y,
            motor_positions.alignment_y,
            md3.alignment_z,
            motor_positions.alignment_z,
            md3.omega,
            motor_positions.omega,
        )


@trace_plan(tracer, "_md3_scan")
def _md3_scan(
    acquisition_uuid: UUID,
    number_of_frames: int,
    scan_range: float,
    exposure_time: float,
    detector_distance: float,
    photon_energy: float,
    transmission: float,
    omega_start: float | None = None,
    number_of_passes: int = 1,
    motor_positions: MotorCoordinates | None = None,
    tray_scan: bool = False,
    count_time: float | None = None,
    hardware_trigger: bool = True,
) -> Generator[Msg, None, None]:
    """
    Runs an MD3 scan on a crystal.

    Parameters
    ----------
    acquisition_uuid : UUID
        The UUID of the acquisition
    number_of_frames : int
        The number of detector frames
    scan_range : float
        The range of the scan in degrees.
    exposure_time : float
        The total exposure time in seconds.
    detector_distance: float
        The detector distance, by default 0.298
    photon_energy : float
        The photon energy in keV, by default 12.7
    transmission : float
        The transmission, must be a value between 0 and 1
    omega_start : float | None, optional
        The start angle of the scan in degrees. If not provided,
        the start angle is inferred depending on the motor positions if the motor
        positions are given. Otherwise, the current omega position is used.
        In case of a tray scan, the start angle is either a) 91 - scan_range/2 or
        b) 270 - scan_range/2 (depending on the tray type)
    number_of_passes : int, optional
        The number of passes, by default 1
    motor_positions : MotorCoordinates | None, optional
        The motor positions at which the scan is done. The motor positions
        usually are inferred by the crystal finder.
    tray_scan : bool, optional
        Determines if the scan is done on a tray, by default False. If tray_scan=True,
        the start angle of the scan is either a) 91 - scan_range/2 or
        b) 270 - scan_range/2 (depending on the tray type)
    count_time : float, optional
        Detector count time. If this parameter is not set, it is set to
        frame_time - 0.0000001 by default. This calculation is done via
        the DetectorConfiguration pydantic model.
    hardware_trigger : bool, optional
        If set to true, we trigger the detector via hardware trigger, by default True.
        Warning! hardware_trigger=False is used mainly for development purposes,
        as it results in a very slow scan

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky stub plan
    """
    if BL_ACTIVE == "false" and tray_scan:
        # Assume a tray type for simulation mode
        yield from mv(md3.omega, 91)

    set_beam_center(detector_distance * 1000)
    md3.save_centring_position()

    yield from set_distance_phase_and_transmission(
        detector_distance * 1000, "DataCollection", transmission
    )

    if omega_start is None:
        omega_start = determine_start_omega(motor_positions, scan_range, tray_scan)

    if motor_positions is not None:
        yield from _move_to_motor_coordinates(motor_positions, tray_scan)

    md3_exposure_time = exposure_time

    frame_rate = number_of_frames / md3_exposure_time

    detector_configuration = DetectorConfiguration(
        roi_mode="disabled",
        trigger_mode="exts",
        nimages=number_of_frames,
        frame_time=1 / frame_rate,
        count_time=count_time,
        ntrigger=number_of_passes,
        user_data=UserData(acquisition_uuid=acquisition_uuid),
        detector_distance=detector_distance,
        photon_energy=photon_energy,
        omega_start=omega_start,
        omega_increment=scan_range / number_of_frames,
    )

    yield from configure(
        dectris_detector,
        detector_configuration.model_dump(
            mode="json", by_alias=True, exclude_none=True
        ),
    )

    yield from stage(dectris_detector)

    save_crystal_pic_to_redis(
        acquisition_uuid=acquisition_uuid,
        collection_stage="start",
    )
    scan_response = yield from _start_md3_scan(
        number_of_frames=number_of_frames,
        omega_start=omega_start,
        scan_range=scan_range,
        md3_exposure_time=md3_exposure_time,
        number_of_passes=number_of_passes,
        tray_scan=tray_scan,
        hardware_trigger=hardware_trigger,
    )

    MD3_SCAN_RESPONSE.put(str(scan_response.model_dump()))
    yield from unstage(dectris_detector)

    save_crystal_pic_to_redis(
        acquisition_uuid=acquisition_uuid,
        collection_stage="end",
    )

    if motor_positions is None:
        # Move back to initial omega position when running mxcube flows
        # If motor_positions is given (UDC), it is not necessary to move back
        yield from mv(md3.omega, omega_start)


@trace_plan(tracer, "md3_scan")
def md3_scan(
    acquisition_uuid: UUID,
    number_of_frames: int,
    scan_range: float,
    exposure_time: float,
    detector_distance: float,
    photon_energy: float,
    transmission: float,
    omega_start: float | None = None,
    number_of_passes: int = 1,
    tray_scan: bool = False,
    motor_positions: MotorCoordinates | None = None,
    count_time: float | None = None,
    hardware_trigger: bool = True,
) -> Generator[Msg, None, None]:
    """
    Runs an MD3 scan on a crystal. If tray_scan=True, the start angle of the scan is either
    a) 91 - scan_range/2 or b) 270 - scan_range/2 (depending on the tray type)

    Parameters
    ----------
    acquisition_uuid : UUID
        The UUID of the acquisition
    number_of_frames : int
        The number of detector frames
    scan_range : float
        The range of the scan in degrees
    exposure_time : float
        The total exposure time in seconds
    detector_distance: float
        The detector distance in meters
    photon_energy: float,
        The photon energy in keV
    transmission : float
        The transmission, must be a value between 0 and 1
    omega_start : float | None, optional
        The start angle of the scan in degrees. If not provided,
        the start angle is inferred depending on the motor positions if the motor
        positions are given. Otherwise, the current omega position is used.
        In case of a tray scan, the start angle is either a) 91 - scan_range/2 or
        b) 270 - scan_range/2 (depending on the tray type)
    number_of_passes : int, optional
        The number of passes, by default 1
    motor_positions : MotorCoordinates | None, optional
        The motor positions at which the scan is done. The motor positions
        usually are inferred by the crystal finder
    tray_scan : bool, optional
        Determines if the scan is done on a tray, by default False
    count_time : float, optional
        Detector count time. If this parameter is not set, it is set to
        frame_time - 0.0000001 by default. This calculation is done via
        the DetectorConfiguration pydantic model.
    hardware_trigger : bool, optional
        If set to true, we trigger the detector via hardware trigger, by default True.
        Warning! hardware_trigger=False is used mainly for development purposes,
        as it results in a very slow scan

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky stub plan
    """

    metadata = {"acquisition_uuid": acquisition_uuid}

    yield from monitor_during_wrapper(
        run_wrapper(
            _md3_scan(
                acquisition_uuid=acquisition_uuid,
                motor_positions=motor_positions,
                number_of_frames=number_of_frames,
                scan_range=scan_range,
                exposure_time=exposure_time,
                number_of_passes=number_of_passes,
                tray_scan=tray_scan,
                count_time=count_time,
                hardware_trigger=hardware_trigger,
                detector_distance=detector_distance,
                photon_energy=photon_energy,
                transmission=transmission,
                omega_start=omega_start,
            ),
            md=metadata,
        ),
        signals=([MD3_SCAN_RESPONSE]),
    )


def _slow_scan(
    start_omega: float, scan_range: float, number_of_frames: int, tray_scan: bool
) -> Generator[Msg, None, MD3ScanResponse]:
    """
    Runs a scan on a crystal. This is a slow scan which triggers the detector
    via software trigger. Note: This plan is intended to be used only for
    development purposes.

    Parameters
    ----------
    start_omega : float
        The initial omega angle
    scan_range : float
        The range of the scan in degrees
    number_of_frames : int
        The detector number of frames
    tray_scan : bool
        Determines if the scan is done on a tray

    Yields
    ------
    Generator[Msg, None, MD3ScanResponse]
        A bluesky plan
    """
    if not tray_scan:
        omega_array = np.linspace(
            start_omega, start_omega + scan_range, number_of_frames
        )
    else:
        omega_position = md3.omega.position
        # There's only two start omega positions depending on the tray type:
        # 91 or 270 degrees. Here, we infer start omega based
        # on the current omega position
        if 70 <= omega_position <= 110:
            yield from mv(md3.omega, 91)
            initial_omega = 91 - scan_range / 2
        elif 250 <= omega_position <= 290:
            yield from mv(md3.omega, 270)
            initial_omega = 270 - scan_range / 2
        else:
            raise ValueError(
                "Start omega should either be in the range (70,110) "
                f"or (250,290). Current value is {omega_position}"
            )
        omega_array = np.linspace(
            initial_omega, initial_omega + scan_range, number_of_frames
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


@trace_plan(tracer, "md3_grid_scan")
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
    detector_distance: float,
    photon_energy: float,
    transmission: float,
    omega_range: float = 0,
    invert_direction: bool = True,
    use_centring_table: bool = True,
    use_fast_mesh_scans: bool = True,
    user_data: UserData | None = None,
    count_time: float | None = None,
    detector_roi_mode: Literal["disabled", "4M"] = "4M",
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
    detector_distance : float
        The detector distance in meters
    photon_energy : float
        The photon energy in keV
    transmission : float
        The transmission, must be a value between 0 and 1
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
    detector_roi_mode : Literal["disabled", "4M"], optional
        ROI mode for the detector, by default "4M"

    Yields
    ------
    Generator
        A bluesky stub plan
    """
    set_beam_center(detector_distance * 1000)

    yield from set_distance_phase_and_transmission(
        detector_distance * 1000, "DataCollection", transmission
    )

    assert number_of_columns > 1, "Number of columns must be > 1"

    frame_rate = number_of_rows / md3_exposure_time

    detector_configuration = DetectorConfiguration(
        roi_mode=detector_roi_mode,
        trigger_mode="exts",
        nimages=number_of_rows,
        frame_time=1 / frame_rate,
        count_time=count_time,
        ntrigger=number_of_columns,
        user_data=user_data,
        detector_distance=detector_distance,
        photon_energy=photon_energy,
        omega_start=md3.omega.position,
        omega_increment=omega_range / (number_of_columns * number_of_rows),
    )

    yield from configure(
        detector,
        detector_configuration.model_dump(
            mode="json", by_alias=True, exclude_none=True
        ),
    )

    yield from stage(detector)

    # Rename variables to make them consistent with MD3 input parameters
    line_range = grid_height
    total_uturn_range = grid_width
    number_of_lines = number_of_columns
    frames_per_lines = number_of_rows

    t = perf_counter()
    # NOTE: The scan_id is stored in the MD3ScanResponse,
    # and is also sent via bluesky documents
    scan_id: int = MD3_CLIENT.startRasterScanEx(
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
    MD3_CLIENT.waitAndCheck(
        task_name="Raster Scan",
        id=scan_id,
        cmd_start=time.perf_counter(),
        expected_time=60,  # TODO: this should be estimated
        timeout=120,  # TODO: this should be estimated
    )
    logger.info(f"Execution time: {perf_counter() - t}")

    task_info = MD3_CLIENT.retrieveTaskInfo(scan_id)

    task_info_model = MD3ScanResponse(
        task_name=task_info[0],
        task_flags=task_info[1],
        start_time=task_info[2],
        end_time=task_info[3],
        task_output=task_info[4],
        task_exception=task_info[5],
        result_id=task_info[6],
    )
    logger.info(f"task info: {task_info_model.model_dump()}")
    if task_info_model.task_exception.lower() != "null":
        raise MD3Exception(
            f"The Scan failed with error {task_info_model.task_exception}. "
            f"Response: {task_info_model.model_dump()}"
        )

    yield from unstage(detector)

    yield from mv(MD3_SCAN_RESPONSE, str(task_info_model.model_dump()))


@trace_plan(tracer, "md3_4d_scan")
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
    detector_distance: float,
    photon_energy: float,
    transmission: float,
    user_data: Optional[UserData] = None,
    count_time: Optional[float] = None,
    detector_roi_mode: Literal["disabled", "4M"] = "4M",
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
    detector_distance : float
        The detector distance in meters
    photon_energy : float
        The photon energy in keV
    transmission : float
        The transmission, must be a value between 0 and 1
    user_data : UserData, optional
        User data pydantic model. This field is passed to the start message
        of the ZMQ stream
    count_time : float, optional
        Detector count time. If this parameter is not set, it is set to
        frame_time - 0.0000001 by default. This calculation is done via
        the DetectorConfiguration pydantic model.
    detector_roi_mode : Literal["disabled", "4M"], optional
        ROI mode for the detector, by default "4M"

    Yields
    ------
    Generator
        A bluesky stub plan
    """
    set_beam_center(detector_distance * 1000)

    yield from set_distance_phase_and_transmission(
        detector_distance * 1000, "DataCollection", transmission
    )

    frame_rate = number_of_frames / md3_exposure_time

    detector_configuration = DetectorConfiguration(
        roi_mode=detector_roi_mode,
        trigger_mode="exts",
        nimages=number_of_frames,
        frame_time=1 / frame_rate,
        count_time=count_time,
        ntrigger=1,
        user_data=user_data,
        detector_distance=detector_distance,
        photon_energy=photon_energy,
        omega_start=md3.omega.position,
        omega_increment=scan_range / number_of_frames,
    )

    yield from configure(
        detector,
        detector_configuration.model_dump(
            mode="json", by_alias=True, exclude_none=True
        ),
    )
    yield from stage(detector)

    # NOTE: The scan_id is stored in the MD3ScanResponse,
    # and is also sent via bluesky documents
    scan_id: int = MD3_CLIENT.startScan4DEx(
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
    MD3_CLIENT.waitAndCheck(
        task_name="Raster Scan",
        id=scan_id,
        cmd_start=time.perf_counter(),
        expected_time=60,  # TODO: this should be estimated
        timeout=120,  # TODO: this should be estimated
    )
    task_info = MD3_CLIENT.retrieveTaskInfo(scan_id)

    task_info_model = MD3ScanResponse(
        task_name=task_info[0],
        task_flags=task_info[1],
        start_time=task_info[2],
        end_time=task_info[3],
        task_output=task_info[4],
        task_exception=task_info[5],
        result_id=task_info[6],
    )

    logger.info(f"task info: {task_info_model.model_dump()}")
    if task_info_model.task_exception.lower() != "null":
        raise MD3Exception(
            f"The Scan failed with error {task_info_model.task_exception}. "
            f"Response: {task_info_model.model_dump()}"
        )
    yield from unstage(detector)

    yield from mv(MD3_SCAN_RESPONSE, str(task_info_model.model_dump()))


def _calculate_alignment_z_motor_coords(
    raster_grid_coords: RasterGridCoordinates,
) -> npt.NDArray:
    # TODO: handle the case when number_of_columns == 1 when using the
    # alignment table. For now, we raise an error
    if raster_grid_coords.number_of_columns == 1:
        raise NotImplementedError(
            "Grid scans with number_of_columns == 1 are not supported "
            "when using the alignment table"
        )

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
    yield from mv(MD3_SCAN_RESPONSE, str(scan_response.model_dump()))
