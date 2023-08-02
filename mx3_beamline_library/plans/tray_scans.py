import logging
import pickle
from os import environ, getcwd, mkdir, path
from time import sleep
from typing import Generator, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import redis
from bluesky.plan_stubs import mv
from bluesky.preprocessors import monitor_during_wrapper, run_wrapper
from bluesky.utils import Msg
from ophyd import Signal

from ..devices.classes.detectors import DectrisDetector
from ..devices.motors import md3
from ..schemas.detector import UserData
from ..schemas.optical_centering import RasterGridCoordinates
from .basic_scans import md3_grid_scan, slow_grid_scan
from .image_analysis import get_image_from_md3_camera, unblur_image
from .plan_stubs import md3_move

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)

REDIS_HOST = environ.get("REDIS_HOST", "0.0.0.0")
REDIS_PORT = int(environ.get("REDIS_PORT", "6379"))
redis_connection = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)

MD3_SCAN_RESPONSE = Signal(name="md3_scan_response", kind="normal")
BL_ACTIVE = environ.get("BL_ACTIVE", "False").lower()


def _single_drop_grid_scan(
    tray_id: str,
    detector: DectrisDetector,
    drop_location: str,
    grid_number_of_columns: int = 15,
    grid_number_of_rows: int = 15,
    exposure_time: float = 1,
    omega_range: float = 0,
    count_time: Optional[float] = None,
    alignment_y_offset: float = 0.2,
    alignment_z_offset: float = -1.0,
    hardware_trigger: bool = True,
) -> Generator[Msg, None, None]:
    """
    Runs a grid-scan on a single drop. If the beamline library is in
    simulation mode, we do a detector software trigger and stream
    grid_number_of_columns*grid_number_of_rows frames, otherwise we use
    the MD3 hardware trigger.

    Parameters
    ----------
    detector : DectrisDetector
        Detector ophyd device
    drop_location : str
        The drop location, e.g. "A1-1"
    grid_number_of_columns : int, optional
        Number of columns of the grid scan, by default 15
    grid_number_of_rows : int, optional
        Number of rows of the grid scan, by default 15
    exposure_time : float, optional
        Exposure time (also know as frame time). NOTE: This is NOT the
        exposure time as defined by the MD3.
    omega_range : float, optional
        Omega range of the grid scan, by default 0
    user_data : UserData, optional
        User data pydantic model. This field is passed to the start message
        of the ZMQ stream, by default None
    count_time : float, optional
        Detector count time. If this parameter is not set, it is set to
        frame_time - 0.0000001 by default. This calculation is done via
        the DetectorConfiguration pydantic model.
    alignment_y_offset : float, optional
        Alignment y offset, determined experimentally, by default 0.2
    alignment_z_offset : float, optional
        Alignment z offset, determined experimentally, by default -1.0
    hardware_trigger : bool, optional
        If set to true, we trigger the detector via hardware trigger, by default True.
        Warning! hardware_trigger=False is used mainly for debugging purposes,
        as it results in a very slow scan

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan
    """

    user_data = UserData(
        id=tray_id,
        zmq_consumer_mode="spotfinder",
        number_of_columns=grid_number_of_columns,
        number_of_rows=grid_number_of_rows,
        grid_scan_id=drop_location,
    )
    assert omega_range <= 10.3, "omega_range must be less that 10.3 degrees"
    # The following seems to be a good approximation of the width of a single drop
    # of the Crystal QuickX2 tray type
    # TODO: support more tray types.
    grid_height = 2.7
    grid_width = 2.7

    md3_exposure_time = grid_number_of_rows * exposure_time

    y_axis_speed = grid_height / md3_exposure_time

    assert y_axis_speed < 15, (
        "grid_height / md3_exposure_time be less than 15 mm/s. The current value is "
        f"{y_axis_speed}. Increase the exposure time. "
    )

    delta_x = grid_width / grid_number_of_columns
    # If grid_width / grid_number_of_columns is too big,
    # the MD3 grid scan does not run successfully
    assert delta_x <= 0.85, (
        "grid_width / grid_number_of_columns must be less than 0.85. "
        f"The current value is {delta_x}. Increase the number of columns"
    )

    if md3.phase.get() != "DataCollection":
        yield from mv(md3.phase, "DataCollection")

    yield from mv(md3.move_plate_to_shelf, drop_location)

    logger.info(f"Plate moved to {drop_location}")

    start_alignment_y = md3.alignment_y.position + alignment_y_offset - grid_height / 2
    # NOTE: 0.86 is the alignment_z default value when
    # the md3 is set to data collection mode
    start_alignment_z = 0.86 + alignment_z_offset - grid_width / 2
    sample_x_position = md3.sample_x.position
    sample_y_position = md3.sample_y.position

    raster_grid_coordinates = RasterGridCoordinates(
        use_centring_table=False,
        initial_pos_sample_x=sample_x_position,
        final_pos_sample_x=sample_x_position,
        initial_pos_sample_y=sample_y_position,
        final_pos_sample_y=sample_y_position,
        initial_pos_alignment_y=start_alignment_y,
        final_pos_alignment_y=start_alignment_y + grid_height,
        initial_pos_alignment_z=start_alignment_z,
        final_pos_alignment_z=start_alignment_z + grid_width,
        width_mm=grid_width,
        height_mm=grid_height,
        number_of_columns=grid_number_of_columns,
        number_of_rows=grid_number_of_rows,
        omega=md3.omega.position,
        alignment_x_pos=md3.alignment_x.position,
        plate_translation=md3.plate_translation.position,
        pixels_per_mm=md3.zoom.pixels_per_mm,
    )
    redis_connection.set(
        f"tray_raster_grid_coordinates_{drop_location}:{user_data.id}",
        pickle.dumps(raster_grid_coordinates.dict()),
    )

    if BL_ACTIVE == "true":
        if hardware_trigger:
            scan_response = yield from md3_grid_scan(
                detector=detector,
                grid_width=grid_width,
                grid_height=grid_height,
                start_omega=md3.omega.position,
                start_alignment_y=start_alignment_y,
                number_of_rows=grid_number_of_rows,
                start_alignment_z=start_alignment_z,
                start_sample_x=md3.sample_x.position,
                start_sample_y=md3.sample_y.position,
                number_of_columns=grid_number_of_columns,
                md3_exposure_time=md3_exposure_time,
                omega_range=omega_range,
                invert_direction=True,
                use_centring_table=False,
                use_fast_mesh_scans=True,
                user_data=user_data,
                count_time=count_time,
            )
        else:
            detector_configuration = {
                "nimages": 1,
                "user_data": user_data.dict(),
                "trigger_mode": "ints",
                "ntrigger": grid_number_of_columns * grid_number_of_rows,
            }

            scan_response = yield from slow_grid_scan(
                raster_grid_coords=raster_grid_coordinates,
                detector=detector,
                detector_configuration=detector_configuration,
                alignment_y=md3.alignment_y,
                alignment_z=md3.alignment_z,
                sample_x=md3.sample_x,
                sample_y=md3.sample_y,
                omega=md3.omega,
                use_centring_table=False,
            )

    elif BL_ACTIVE == "false":
        # Do a software trigger and return a random scan response
        detector_configuration = {
            "nimages": 1,
            "user_data": user_data.dict(),
            "trigger_mode": "ints",
            "ntrigger": grid_number_of_columns * grid_number_of_rows,
        }

        scan_response = yield from slow_grid_scan(
            raster_grid_coords=raster_grid_coordinates,
            detector=detector,
            detector_configuration=detector_configuration,
            alignment_y=md3.alignment_y,
            alignment_z=md3.alignment_z,
            sample_x=md3.sample_x,
            sample_y=md3.sample_y,
            omega=md3.omega,
            use_centring_table=False,
        )

    MD3_SCAN_RESPONSE.put(str(scan_response.dict()))

    if scan_response.task_exception.lower() != "null":
        raise RuntimeError(
            f"Grid scan did not run successfully: {scan_response.dict()}"
        )
    return scan_response


def single_drop_grid_scan(
    tray_id: str,
    detector: DectrisDetector,
    drop_location: str,
    grid_number_of_columns: int = 15,
    grid_number_of_rows: int = 15,
    exposure_time: float = 1,
    omega_range: float = 0,
    count_time: Optional[float] = None,
    alignment_y_offset: float = 0.2,
    alignment_z_offset: float = -1.0,
    hardware_trigger: bool = True,
) -> Generator[Msg, None, None]:
    """
    Wrapper of the _single_drop_grid_scan function. This allows us to
    send relevant metadata through kafka

    Parameters
    ----------
    detector : DectrisDetector
        Detector ophyd device
    drop_location : str
        The drop location, e.g. "A1-1"
    grid_number_of_columns : int, optional
        Number of columns of the grid scan, by default 15
    grid_number_of_rows : int, optional
        Number of rows of the grid scan, by default 15
    exposure_time : float, optional
        Exposure time (also know as frame time). NOTE: This is NOT the
        exposure time as defined by the MD3.
    omega_range : float, optional
        Omega range of the grid scan, by default 0
    user_data : UserData, optional
        User data pydantic model. This field is passed to the start message
        of the ZMQ stream, by default None
    count_time : float, optional
        Detector count time. If this parameter is not set, it is set to
        frame_time - 0.0000001 by default. This calculation is done via
        the DetectorConfiguration pydantic model.
    alignment_y_offset : float, optional
        Alignment y offset, determined experimentally, by default 0.2
    alignment_z_offset : float, optional
        Alignment z offset, determined experimentally, by default -1.0
    hardware_trigger : bool, optional
        If set to true, we trigger the detector via hardware trigger, by default True.
        Warning! hardware_trigger=False is used mainly for debugging purposes,
        as it results in a very slow scan

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan
    """

    yield from monitor_during_wrapper(
        run_wrapper(
            _single_drop_grid_scan(
                tray_id=tray_id,
                detector=detector,
                drop_location=drop_location,
                grid_number_of_columns=grid_number_of_columns,
                grid_number_of_rows=grid_number_of_rows,
                exposure_time=exposure_time,
                omega_range=omega_range,
                count_time=count_time,
                alignment_y_offset=alignment_y_offset,
                alignment_z_offset=alignment_z_offset,
                hardware_trigger=hardware_trigger,
            ),
            md={"tray_id": tray_id, "drop_location": drop_location},
        ),
        signals=([MD3_SCAN_RESPONSE]),
    )


def multiple_drop_grid_scan(
    tray_id: str,
    detector: DectrisDetector,
    drop_locations: list[str],
    grid_number_of_columns: int = 15,
    grid_number_of_rows: int = 15,
    exposure_time: float = 1,
    omega_range: float = 0,
    count_time: Optional[float] = None,
    alignment_y_offset: float = 0.2,
    alignment_z_offset: float = -1.0,
    hardware_trigger: bool = True,
) -> Generator[Msg, None, None]:
    """
    Runs one grid scan per drop. The drop locations are specified in the
    drop_locations argument, e.g. drop_locations=["A1-1", "A1-2"]

    Parameters
    ----------
    tray_id: str
        The id of the tray
    detector : DectrisDetector
        Detector ophyd device
    drop_locations : list[str]
        A list of drop locations, e.g. ["A1-1", "A1-2"]
    grid_number_of_columns : int, optional
        Number of columns of the grid scan, by default 15
    grid_number_of_rows : int, optional
        Number of rows of the grid scan, by default 15
    exposure_time : float
        Exposure time measured in seconds to control shutter command. Note that
        this is the exposure time of one column only, e.g. the md3 takes
        `exposure_time` seconds to move `grid_height` mm.
    omega_range : float, optional
        Omega range of the grid scan, by default 0
    user_data : Union[UserData, dict], optional
        User data pydantic model, or dictionary. This field is passed to the start
        message of the ZMQ stream, by default None.
        Note that we also support a dictionary because the bluesky
        queueserver can't handle pydantic models as input data
    count_time : float, optional
        Detector count time. If this parameter is not set, it is set to
        frame_time - 0.0000001 by default. This calculation is done via
        the DetectorConfiguration pydantic model.
    alignment_y_offset : float, optional
        Alignment y offset, determined experimentally, by default 0.2
    alignment_z_offset : float, optional
        ALignment z offset, determined experimentally, by default -1.0
    hardware_trigger : bool, optional
        If set to true, we trigger the detector via hardware trigger, by default True.
        Warning! hardware_trigger=False is used mainly for debugging purposes,
        as it results in a very slow scan

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan
    """

    drop_locations.sort()  # sort list to scan drops faster
    for drop in drop_locations:
        yield from single_drop_grid_scan(
            tray_id=tray_id,
            detector=detector,
            drop_location=drop,
            grid_number_of_columns=grid_number_of_columns,
            grid_number_of_rows=grid_number_of_rows,
            exposure_time=exposure_time,
            omega_range=omega_range,
            count_time=count_time,
            alignment_y_offset=alignment_y_offset,
            alignment_z_offset=alignment_z_offset,
            hardware_trigger=hardware_trigger,
        )


def save_drop_snapshots(
    tray_id: str,
    drop_locations: list[str],
    alignment_y_offset: float = 0.25,
    alignment_z_offset: float = -1.0,
    min_focus: float = -1.0,
    max_focus: float = 0.5,
    tol: float = 0.1,
    number_of_intervals: int = 2,
    output_directory: Optional[str] = None,
    backlight_value: Optional[float] = None,
) -> Generator[Msg, None, None]:
    """
    This plan takes drop snapshots at the positions specified by the
    drop_location list. The results are saved to the output_directory.
    Whenever we move the md3 to a new drop location, we unblur the image.

    Parameters
    ----------
    tray_id : str
        The id of the tray
    drop_locations : list[str]
        A list of drop locations, e.g ["A1-1", "B2-2"]
    alignment_y_offset : float, optional
        Alignment y offset, by default 0.25
    alignment_z_offset : float, optional
        Alignment z offset, by default -1.0
    min_focus : float, optional
        Minimum value to search for the maximum of var( Img * L(x,y) ),
        by default -1.0
    max_focus : float, optional
        Maximum value to search for the maximum of var( Img * L(x,y) ),
        by default 0.5
    tol : float, optional
        The tolerance used by the Golden-section search, by default 0.1
    number_of_intervals : int, optional
        Number of intervals used to find local maximums of the function
        `var( Img * L(x,y) )`, by default 2
    output_directory : Optional[str],
        The output directory. If output_directory=None, the results are
        saved to the current working directory, by default None
    backlight_value : Optional[float]
        Backlight value

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan
    """
    if output_directory is None:
        output_directory = getcwd()

    try:
        mkdir(path.join(output_directory, tray_id))
    except FileExistsError:
        logger.info("Folder exists. Overwriting results")

    drop_locations.sort()  # sort list to scan drops faster

    if backlight_value is not None:
        md3.backlight.set(backlight_value)

    for drop in drop_locations:
        yield from mv(md3.move_plate_to_shelf, drop)

        sleep(1)
        start_alignment_y = md3.alignment_y.position + alignment_y_offset
        start_alignment_z = md3.alignment_z.position + alignment_z_offset

        yield from md3_move(
            md3.alignment_y, start_alignment_y, md3.alignment_z, start_alignment_z
        )
        yield from unblur_image(
            md3.alignment_x, min_focus, max_focus, tol, number_of_intervals
        )

        _path = path.join(output_directory, tray_id, drop)
        plt.figure()
        plt.imshow(get_image_from_md3_camera(np.uint16))
        plt.savefig(_path)
        plt.close()

        motor_positions = {
            "sample_x": md3.sample_x.position,
            "sample_y": md3.sample_y.position,
            "alignment_x": md3.alignment_x.position,
            "alignment_y": md3.alignment_y.position,
            "alignment_z": md3.alignment_z.position,
            "backlight": md3.backlight.get(),
            "omega": md3.omega.position,
            "plate_translation": md3.plate_translation.position,
        }
        df = pd.DataFrame(motor_positions, index=[0])
        df.to_csv(f"{_path}.csv", index=False)


def save_drop_snapshots_from_motor_positions(
    tray_id: str,
    drop_list: list[str],
    input_directory: str,
    plate_translation_offset: float = 0,
    output_directory: Optional[str] = None,
) -> Generator[Msg, None, None]:
    """
    This plan saves drop snapshots from motor positions saved by the save_drop_snapshots plan.

    Parameters
    ----------
    tray_id : str
        The tray id
    drop_list : list[str]
        A list of drops
    input_directory : str
        The input directory where the motor positions are saved. This folder is created by the
        save_drop_snapshots plan, e.g. /mnt/shares/smd_share/tray_images/tray_1
    plate_translation_calibration : float, optional
        Calibration of the plate translation
    output_directory : Optional[str], optional
        The output directory, by default None

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plans
    """

    if output_directory is None:
        output_directory = getcwd()

    try:
        mkdir(path.join(output_directory, tray_id))
    except FileExistsError:
        logger.info("Folder exists. Overwriting results")

    for drop in drop_list:
        df = pd.read_csv(path.join(input_directory, f"{drop}.csv"))
        yield from md3_move(
            md3.sample_x,
            df["sample_x"][0],
            md3.sample_y,
            df["sample_y"][0],
            md3.alignment_x,
            df["alignment_x"][0],
            md3.alignment_y,
            df["alignment_y"][0],
            md3.alignment_z,
            df["alignment_z"][0] + plate_translation_offset,
            md3.plate_translation,
            df["plate_translation"][0],
        )

        _path = path.join(output_directory, tray_id, drop)
        plt.figure()
        plt.imshow(get_image_from_md3_camera(np.uint16))
        plt.savefig(_path)
        plt.close()
