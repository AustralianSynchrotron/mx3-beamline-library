import logging
from typing import Generator, Optional

from bluesky.plan_stubs import configure, mv, stage, trigger_and_read, unstage  # noqa
from bluesky.utils import Msg

from ..devices.classes.detectors import DectrisDetector
from ..devices.motors import md3
from ..schemas.detector import UserData
from .basic_scans import md3_grid_scan

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)


def single_drop_grid_scan(
    detector: DectrisDetector,
    column: int,
    row: int,
    drop: int,
    grid_number_of_columns: int = 15,
    grid_number_of_rows: int = 15,
    exposure_time: float = 1,
    user_data: Optional[UserData] = None,
    count_time: Optional[float] = None,
    alignment_y_offset: float = 0.2,
    alignment_z_offset: float = -1.0,
) -> Generator[Msg, None, None]:
    """
    Runs a grid-scan on a single drop.

    Parameters
    ----------
    detector : DectrisDetector
        Detector ophyd device
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
    alignment_y_offset : float, optional
        Alignment y offset, determined experimentally, by default 0.2
    alignment_z_offset : float, optional
        ALignment z offset, determined experimentally, by default -1.0

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan
    """
    assert 0 <= column <= 11, "Column must be a number between 0 and 11"
    assert 0 <= row <= 7, "Row must be a number between 0 and 7"
    assert 0 <= drop <= 2, "Drop must be a number between 0 and 2"
    # The following seems to be a good approximation of the width of a single drop
    # of the Crystal QuickX2 tray type
    grid_height = 3.4
    grid_width = 3.4

    delta_x = grid_width / grid_number_of_columns
    # If grid_width / grid_number_of_columns is too big,
    # the MD3 grid scan does not run successfully
    assert delta_x <= 0.85, (
        "grid_width / grid_number_of_columns <= 0.85. "
        + "Increase the number of columns"
    )

    if user_data is not None:
        user_data.number_of_columns = grid_number_of_columns
        user_data.number_of_rows = grid_number_of_rows

    if md3.phase.get() != "DataCollection":
        yield from mv(md3.phase, "DataCollection")

    yield from mv(md3.move_plate_to_shelf, (row, column, drop))

    logger.info(f"Plate moved to ({row}, {column}, {drop})")

    start_alignment_y = md3.alignment_y.position + alignment_y_offset - grid_height / 2
    start_alignment_z = md3.alignment_z.position + alignment_z_offset - grid_width / 2

    yield from md3_grid_scan(
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
        exposure_time=exposure_time,
        omega_range=0,
        invert_direction=True,
        use_centring_table=False,
        use_fast_mesh_scans=True,
        user_data=user_data,
        count_time=count_time,
    )


def multiple_drop_grid_scan(
    detector: DectrisDetector,
    drop_locations: list[str],
    grid_number_of_columns: int = 15,
    grid_number_of_rows: int = 15,
    exposure_time: float = 1,
    user_data: Optional[UserData] = None,
    count_time: Optional[float] = None,
    alignment_y_offset: float = 0.2,
    alignment_z_offset: float = -1.0,
) -> Generator[Msg, None, None]:
    """
    Runs one grid scan per drop. The drop locations are specified in the
    drop_locations argument, e.g. drop_locations=["A1-1", "A1-2"]

    Parameters
    ----------
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
        ALignment z offset, determined experimentally, by default -1.0

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan
    """

    drop_locations.sort()  # sort list for efficiency
    for drop in drop_locations:
        if user_data is not None:
            user_data.drop_location = drop

        assert (
            len(drop) == 4
        ), "The drop location should follow a format similar to e.g. A1-1"

        row = ord(drop[0].upper()) - 65  # This converts letters to numbers e.g. A=0
        column = int(drop[1]) - 1  # We count from 0, not 1
        assert (
            drop[2] == "-"
        ), "The drop location should follow a format similar to e.g. A1-1"
        drop = int(drop[3]) - 1  # We count from 0, not 1

        yield from single_drop_grid_scan(
            detector=detector,
            column=column,
            row=row,
            drop=drop,
            grid_number_of_columns=grid_number_of_columns,
            grid_number_of_rows=grid_number_of_rows,
            exposure_time=exposure_time,
            user_data=user_data,
            count_time=count_time,
            alignment_y_offset=alignment_y_offset,
            alignment_z_offset=alignment_z_offset,
        )
