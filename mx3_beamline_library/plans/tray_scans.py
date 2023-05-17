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


def scan_drop(
        detector: DectrisDetector,
        column:int , row: int, drop: int, grid_number_of_columns: int = 15, 
        grid_number_of_rows: int = 15, exposure_time: float = 1,
        user_data: Optional[UserData] = None,
        count_time: Optional[float] = None):
    """
    Scans a drop given a colum, row, and drop id.
    The number of columns and rows of the grid, as well as the exposure time
    can also be specified

    Parameters
    ----------
    column : int
        Column of the cell (0 to 11).
    row : int
        Row of the cell (0 to 7),containing one or several shelves/drops.
    drop : int
        Drop index (Starting from 0).
    """

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


    SERVER.movePlateToShelf(row, column, drop)
    logger.info(f"Plate successfully moved to ({row}, {column})")

    delta = 1.7 # mm


    grid_height = delta*2
    grid_width = delta*2
    logger.info(f"delta_x: {grid_width / grid_number_of_columns}")



    start_omega =  md3.omega.position
    omega_range = 0

    start_alignment_y = md3.alignment_y.position - delta
    start_alignment_z = md3.alignment_z.position - delta
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
        