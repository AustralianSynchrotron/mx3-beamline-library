import logging
from os import environ
from typing import Generator, Optional

from bluesky.plan_stubs import configure, mv, stage, trigger_and_read, unstage  # noqa
from bluesky.utils import Msg

from ..devices.classes.detectors import DectrisDetector
from ..devices.motors import md3
from ..schemas.detector import UserData
from ..schemas.xray_centering import MD3ScanResponse
from .basic_scans import arm_trigger_and_disarm_detector, md3_grid_scan


from mx3_beamline_library.schemas.detector import UserData
from bluesky.plan_stubs import mv
import numpy as np
import cv2
from mx3_beamline_library.devices.detectors import md_camera  # noqa
from time import sleep
import matplotlib.pyplot as plt
from mx3_beamline_library.devices.motors import md3


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
    omega_range: float = 0,
    user_data: Optional[UserData] = None,
    count_time: Optional[float] = None,
    alignment_y_offset: float = 0.2,
    alignment_z_offset: float = -1.0,
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
    column : int
        Column of the cell (0 to 11).
    row : int
        Row of the cell (0 to 7), containing one or several shelves/drops.
    drop : int
        Drop index (0 to 3).
    grid_number_of_columns : int, optional
        Number of columns of the grid scan, by default 15
    grid_number_of_rows : int, optional
        Number of rows of the grid scan, by default 15
    exposure_time : float, optional
        Exposure time measured in seconds to control shutter command. Note that
        this is the exposure time of one column only, e.g. the md3 takes
        `exposure_time` seconds to move `grid_height` mm.
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
        ALignment z offset, determined experimentally, by default -1.0

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan
    """
    assert 0 <= column <= 11, "Column must be a number between 0 and 11"
    assert 0 <= row <= 7, "Row must be a number between 0 and 7"
    assert 0 <= drop <= 2, "Drop must be a number between 0 and 2"
    assert omega_range <= 10.3, "omega_range must be less that 10.3 degrees"
    # The following seems to be a good approximation of the width of a single drop
    # of the Crystal QuickX2 tray type
    # TODO: support more tray types.
    grid_height = 3.4
    grid_width = 3.4

    y_axis_speed = grid_height / exposure_time

    assert y_axis_speed < 5.7, (
        "grid_height / exposure_time be less than 5.7 mm/s. The current value is "
        f"{y_axis_speed}. Increase the exposure time. "
        "NOTE: The 5.7 mm/s value was calculated experimentally, so this value "
        "may not be completely accurate."
    )

    delta_x = grid_width / grid_number_of_columns
    # If grid_width / grid_number_of_columns is too big,
    # the MD3 grid scan does not run successfully
    assert delta_x <= 0.85, (
        "grid_width / grid_number_of_columns must be less than 0.85. "
        f"The current value is {delta_x}. Increase the number of columns"
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

    if environ["BL_ACTIVE"].lower() == "true":
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
            exposure_time=exposure_time,
            omega_range=omega_range,
            invert_direction=True,
            use_centring_table=False,
            use_fast_mesh_scans=True,
            user_data=user_data,
            count_time=count_time,
        )
    elif environ["BL_ACTIVE"].lower() == "false":
        # Do a software trigger and return a random scan response
        detector_configuration = {
            "nimages": grid_number_of_columns * grid_number_of_rows,
            "user_data": user_data.dict(),
        }
        yield from arm_trigger_and_disarm_detector(
            detector=detector,
            detector_configuration=detector_configuration,
            metadata={},
        )
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


def multiple_drop_grid_scan(
    detector: DectrisDetector,
    drop_locations: list[str],
    grid_number_of_columns: int = 15,
    grid_number_of_rows: int = 15,
    exposure_time: float = 1,
    omega_range: float = 0,
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
        ALignment z offset, determined experimentally, by default -1.0

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan
    """

    drop_locations.sort()  # sort list to scan drops faster
    for drop in drop_locations:
        if user_data is not None:
            user_data.grid_scan_id = drop

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
            omega_range=omega_range,
            user_data=user_data,
            count_time=count_time,
            alignment_y_offset=alignment_y_offset,
            alignment_z_offset=alignment_z_offset,
        )







def unblur_image(
    focus_motor,
    a: float = 0.0,
    b: float = 1.0,
    tol: float = 0.2,
    number_of_intervals: int = 2,
):
    """
    We use the Golden-section search to find the global maximum of the variance function
    described in the calculate_variance method ( `var( Img * L(x,y) )` )
    (see the definition of self.variance_local_maximum).
    In order to find the global maximum, we search for local maximums in N number of
    sub-intervals defined by number_of_intervals.

    Parameters
    ----------
    motor : MD3Motor
        An MD3 motor. We can focus the image with either alignment x, or sample_x and
        sample_y (depending on the value of omega)
    a : float, optional
        Minimum value to search for the maximum of var( Img * L(x,y) )
    b : float, optional
        Maximum value to search for the maximum of var( Img * L(x,y) )
    tol : float, optional
        The tolerance, by default 0.2
    number_of_intervals : int, optional
        Number of sub-intervals used to find the global maximum of a multimodal function

    Yields
    ------
    Generator[Msg, None, None]
        Moves the focus motor to a position where the image is focused
    """

    # Create sub-intervals to find the global maximum
    step = (b - a) / number_of_intervals
    interval_list = []
    for i in range(number_of_intervals):
        interval_list.append((a + step * i, a + step * (i + 1)))

    # Calculate local maximums
    laplacian_list = []
    focus_motor_pos_list = []
    for interval in interval_list:
        yield from variance_local_maximum(
            focus_motor, interval[0], interval[1], tol
        )
        laplacian_list.append(calculate_variance())
        focus_motor_pos_list.append(focus_motor.position)

    # Find global maximum, and move the focus motor to the best focused position
    argmax = np.argmax(np.array(laplacian_list))
    yield from mv(focus_motor, focus_motor_pos_list[argmax])

def variance_local_maximum(
    focus_motor,
    a: float = 0.0,
    b: float = 1.0,
    tol: float = 0.2,
) -> float:
    """
    We use the Golden-section search to find the local maximum of the variance function
    described in the calculate_variance method ( `var( Img * L(x,y) )` ).
    NOTE: We assume that the function is strictly unimodal on [a,b].
    See for example: https://en.wikipedia.org/wiki/Golden-section_search

    Parameters
    ----------
    focus_motor : MD3Motor
        An MD3 motor, can be either a combination of sample_x and
        sample_y, or alignment_x
    a : float
        Minimum value to search for the maximum of var( Img * L(x,y) )
    b : float
        Maximum value to search for the maximum of var( Img * L(x,y) )
    tol : float, optional
        The tolerance, by default 0.2

    Returns
    -------
    Generator[Msg, None, None]
        Moves sample_y to a position where the image is focused
    """
    gr = (np.sqrt(5) + 1) / 2

    c = b - (b - a) / gr
    d = a + (b - a) / gr

    count = 0
    while abs(b - a) > tol:
        yield from mv(focus_motor, c)
        val_c = calculate_variance()

        yield from mv(focus_motor, d)
        val_d = calculate_variance()

        if val_c > val_d:  # val_c > val_d to find the maximum
            b = d
        else:
            a = c

        # We recompute both c and d here to avoid loss of precision which
        # may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        count += 1

    maximum = (b + a) / 2
    yield from mv(focus_motor, maximum)

def calculate_variance() -> float:
    """
    We calculate the variance of the convolution of the laplacian kernel with an image,
    e.g. var( Img * L(x,y) ), where Img is an image taken from the camera ophyd object,
    and L(x,y) is the Laplacian kernel.

    Returns
    -------
    float
        var( Img * L(x,y) )
    """
    data = get_image_from_md3_camera(np.uint8)

    try:
        gray_image = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
    except cv2.error:
        # The MD3 camera already returns black and white images for the zoom levels
        # 5, 6 and 7, so we don't do anything here
        gray_image = data

    return cv2.Laplacian(gray_image, cv2.CV_64F).var()

def get_image_from_md3_camera( dtype):
    """
    Gets a frame from the md3.

    Parameters
    ----------
    dtype : npt.DTypeLike, optional
        The data type of the numpy array, by default np.uint16

    Returns
    -------
    npt.NDArray
        A frame of shape (height, width, depth)
    """
    if environ["BL_ACTIVE"].lower() == "true":
        array_data = md_camera.array_data.get()
        data = array_data.astype(dtype)
    else:
        # When the camera is not working, we stream a static image
        # of the test rig
        data = (
            np.load("/mnt/shares/smd_share/blackfly_cam_images/flat.npy")
            .astype(dtype)
            .astype(dtype)
        )
    return data


def get_drop_images(drop_locations: list[str], alignment_y_offset, alignment_z_offset):
    drop_locations.sort()  # sort list to scan drops faster
    for drop in drop_locations:

        assert (
            len(drop) == 4
        ), "The drop location should follow a format similar to e.g. A1-1"

        row = ord(drop[0].upper()) - 65  # This converts letters to numbers e.g. A=0
        column = int(drop[1]) - 1  # We count from 0, not 1
        assert (
            drop[2] == "-"
        ), "The drop location should follow a format similar to e.g. A1-1"
        _drop = int(drop[3]) - 1  # We count from 0, not 1
        yield from mv(md3.move_plate_to_shelf, (row, column, _drop) )

        sleep(1)
        start_alignment_y = md3.alignment_y.position + alignment_y_offset
        start_alignment_z = md3.alignment_z.position + alignment_z_offset

        yield from mv(md3.alignment_y, start_alignment_y)
        yield from mv( md3.alignment_z, start_alignment_z)
        sleep(1)
        yield from unblur_image(md3.alignment_x, -1, 0.5, 0.1, number_of_intervals=2)
        
        plt.figure()
        plt.imshow(get_image_from_md3_camera(np.uint16))
        plt.savefig(drop)
        plt.close()
