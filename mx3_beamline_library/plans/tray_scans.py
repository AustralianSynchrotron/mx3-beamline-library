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
from ..devices.classes.motors import SERVER, MD3Motor
from ..schemas.detector import DetectorConfiguration, UserData
from ..schemas.xray_centering import MD3ScanResponse, RasterGridCoordinates
from ..devices.motors import md3

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)


def single_drop_grid_scan(
        detector: DectrisDetector,
        column:int , row: int, drop: int, grid_number_of_columns: int = 15, 
        grid_number_of_rows: int = 15, exposure_time: float = 1,
        user_data: Optional[UserData] = None,
        count_time: Optional[float] = None, alignment_z_offset: float = -1.0
        ) -> Generator[Msg, None, None]:
    """
    Runs a grid-scan on a single drop.

    Parameters
    ----------
    column : int
        Column of the cell (0 to 11).
    row : int
        Row of the cell (0 to 7),containing one or several shelves/drops.
    drop : int
        Drop index (Starting from 0).
    grid_number_of_columns : int, optional
        Number of columns of the grid scan, by default 15
    grid_number_of_rows : int, optional
        Number of rows of the grid scan, by default 15
    exposure_time : float
        Exposure time measured in seconds to control shutter command. Note that
        this is the exposure time of one column only, e.g. the md3 takes
        `exposure_time` seconds to move `grid_height` mm.
    user_data : UserData, optional
        User data pydantic model. This field is passed to the start message
        of the ZMQ stream, by default None
    count_time : float, optional
        Detector count time. If this parameter is not set, it is set to
        frame_time - 0.0000001 by default. This calculation is done via
        the DetectorConfiguration pydantic model.

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan
    """
    assert 0 <= column <= 11, "Column must be a number between 0 and 11"
    assert 0 <= row <= 7, "Row must be a number between 0 and 7"
    # The following seems to be a good approximation of the width of a single drop 
    # of theCrystal QuickX2 tray type
    grid_height = 3.4
    grid_width = 3.4

    delta_x = grid_width / grid_number_of_columns
    # If the jump of the grid scan in the x axis is too big
    # the MD3 grid scan does not run successfully
    assert delta_x <= 0.85, "grid_width / grid_number_of_columns <= 0.85. " + \
        "Increase the number of columns"

    if user_data is not None:
        user_data.number_of_columns = grid_number_of_columns
        user_data.number_of_rows = grid_number_of_rows

    frame_rate = grid_number_of_rows / exposure_time

    detector_configuration = DetectorConfiguration(
        trigger_mode="exts",
        nimages=grid_number_of_rows,
        frame_time=1 / frame_rate,
        count_time=count_time,
        ntrigger=grid_number_of_columns,
        user_data=user_data,
    )


    if md3.phase.get() != "DataCollection":
        yield from mv(md3.phase, "DataCollection")

    yield from configure(detector, detector_configuration.dict(exclude_none=True))

    yield from stage(detector)

    yield from mv(md3.move_plate_to_shelf, (row, column, drop) )

    logger.info(f"Plate successfully moved to ({row}, {column}, {drop})")

    start_omega =  md3.omega.position
    omega_range = 0

    start_alignment_y = md3.alignment_y.position - grid_height / 2
    start_alignment_z = md3.alignment_z.position + alignment_z_offset - grid_width / 2
    start_sample_x = md3.sample_x.position
    start_sample_y = md3.sample_y.position
    invert_direction = True
    use_centring_table = False
    use_fast_mesh_scans = True


    line_range = grid_height
    total_uturn_range = grid_width
    number_of_lines = grid_number_of_columns
    frames_per_lines = grid_number_of_rows

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


def multiple_drop_grid_scans(detector: DectrisDetector, drop_locations: list[str],
        grid_number_of_columns: int = 15, 
        grid_number_of_rows: int = 15, exposure_time: float = 1,
        user_data: Optional[UserData] = None,
        count_time: Optional[float] = None,
        alignment_z_offset: float = -1.0) -> Generator[Msg, None, None]:
    """
    Runs one grid scan per drop.

    Parameters
    ----------
    detector : DectrisDetector
        _description_
    drop_locations : list[str]
        A list of drop locations, e.g. ["A1-1", "A1-2"]
    grid_number_of_columns : int, optional
        _description_, by default 15
    grid_number_of_columns : int, optional
        Number of columns of the grid scan, by default 15
    grid_number_of_rows : int, optional
        Number of rows of the grid scan, by default 15
    exposure_time : float
        Exposure time measured in seconds to control shutter command. Note that
        this is the exposure time of one column only, e.g. the md3 takes
        `exposure_time` seconds to move `grid_height` mm.
    user_data : UserData, optional
        User data pydantic model. This field is passed to the start message
        of the ZMQ stream, by default None
    count_time : float, optional
        Detector count time. If this parameter is not set, it is set to
        frame_time - 0.0000001 by default. This calculation is done via
        the DetectorConfiguration pydantic model.

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan
    """
    
    drop_locations.sort() # sort list for efficiency
    for drop in drop_locations:
        column = ord(drop[0].upper()) - 65 # This converts letters to numbers e.g. A=0
        row = int(drop[1]) - 1 # we like to count starting from zero, not 1
        drop = int(drop[3]) - 1 # same here
        
        yield from single_drop_grid_scan(
            detector=detector, column=column, row=row, drop=drop,
            grid_number_of_columns=grid_number_of_columns, grid_number_of_rows=grid_number_of_rows,
            exposure_time=exposure_time,
            user_data=user_data, count_time=count_time, alignment_z_offset=alignment_z_offset
            )

