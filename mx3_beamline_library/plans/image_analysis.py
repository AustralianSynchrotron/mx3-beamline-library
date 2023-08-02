import logging
from os import environ
from typing import Generator

import cv2
import numpy as np
import numpy.typing as npt
from bluesky.plan_stubs import mv
from bluesky.utils import Msg

from ..devices.classes.motors import MD3Motor
from ..devices.detectors import blackfly_camera, md_camera
from ..devices.sim.classes.detectors import SIM_MD3_CAMERA_IMG, SIM_TOP_CAMERA_IMG

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)


def unblur_image_fast(
    focus_motor: MD3Motor, start_position=-0.2, final_position=1.3
) -> Generator[Msg, None, float]:
    """
    This method unblurs an image by continuously taking snapshots from the
    md3 camera while moving the focus motor (usually alignment_x)
    between start_position and final_position

    Parameters
    ----------
    focus_motor : MD3Motor
        The motor used to focus a sample, usually alignment_x
    start_position : float, optional
        The start position of focus motor, by default -0.2
    final_position : float, optional
        The final position of focus motor, by default 1.3

    Yields
    ------
    Generator[Msg, None, float]
         A bluesky plan
    """
    yield from mv(focus_motor, start_position)
    yield from mv(focus_motor, final_position, wait=False)
    while not focus_motor.moving:
        pass

    variance_list = []
    alignment_x_positions = []
    while focus_motor.moving:
        variance_list.append(_calculate_variance())
        alignment_x_positions.append(focus_motor.position)

    focused_position = alignment_x_positions[np.argmax(variance_list)]
    return focused_position


def unblur_image(
    focus_motor: MD3Motor,
    a: float = 0.0,
    b: float = 1.0,
    tol: float = 0.2,
    number_of_intervals: int = 2,
) -> Generator[Msg, None, None]:
    """
    We use the Golden-section search to find the global maximum of the variance function
    described in the _calculate_variance method ( `var( Img * L(x,y) )` )
    (see the definition of self._variance_local_maximum).
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
        yield from _variance_local_maximum(focus_motor, interval[0], interval[1], tol)
        laplacian_list.append(_calculate_variance())
        focus_motor_pos_list.append(focus_motor.position)

    # Find global maximum, and move the focus motor to the best focused position
    argmax = np.argmax(np.array(laplacian_list))
    yield from mv(focus_motor, focus_motor_pos_list[argmax])


def _variance_local_maximum(
    focus_motor,
    a: float = 0.0,
    b: float = 1.0,
    tol: float = 0.2,
) -> float:
    """
    We use the Golden-section search to find the local maximum of the variance function
    described in the _calculate_variance method ( `var( Img * L(x,y) )` ).
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
    logger.info("Focusing image...")
    while abs(b - a) > tol:
        yield from mv(focus_motor, c)
        val_c = _calculate_variance()

        yield from mv(focus_motor, d)
        val_d = _calculate_variance()

        if val_c > val_d:  # val_c > val_d to find the maximum
            b = d
        else:
            a = c

        # We recompute both c and d here to avoid loss of precision which
        # may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        count += 1
        logger.info(f"Iteration: {count}")
    maximum = (b + a) / 2
    logger.info(f"Optimal value: {maximum}")
    yield from mv(focus_motor, maximum)


def _calculate_variance() -> float:
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


def get_image_from_md3_camera(dtype: npt.DTypeLike = np.uint16) -> npt.NDArray:
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
        array_data: npt.NDArray = md_camera.array_data.get()
        data = array_data.astype(dtype)
    else:
        # When the camera is not working, we stream a static image
        # of the test rig
        data = SIM_MD3_CAMERA_IMG.astype(dtype)
    return data


def get_image_from_top_camera(
    dtype: npt.DTypeLike = np.uint16,
) -> tuple[npt.NDArray, int, int]:
    """
    Gets a frame from the top camera. Since the returned frame is a flattened image,
    we also return the width and height

    Parameters
    ----------
    dtype : npt.DTypeLike, optional
        The data type of the numpy array, by default np.uint16

    Returns
    -------
    tuple[npt.NDArray, int, int]
        A flattened image, the height, and the width
    """
    if environ["BL_ACTIVE"].lower() == "true":
        array_data: npt.NDArray = blackfly_camera.array_data.get()
        image = array_data.astype(dtype)
        height = blackfly_camera.height.get()
        width = blackfly_camera.width.get()
    else:
        # When the camera is not working, we stream a static image
        image = SIM_TOP_CAMERA_IMG.astype(dtype)
        height = 1024
        width = 1224

    return image, height, width
