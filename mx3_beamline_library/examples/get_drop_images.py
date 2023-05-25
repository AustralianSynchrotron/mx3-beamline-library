
from os import environ

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback

environ["BL_ACTIVE"] = "True"
environ["MD3_ADDRESS"] = "10.244.101.30"
environ["MD3_PORT"] = "9001"
environ["MD_REDIS_HOST"] = "10.244.101.30"
environ["MD_REDIS_PORT"] = "6379"

from mx3_beamline_library.devices.detectors import dectris_detector
from mx3_beamline_library.plans.tray_scans import multiple_drop_grid_scan
from mx3_beamline_library.schemas.detector import UserData
from bluesky.plan_stubs import configure, mv
import numpy as np
import cv2
from mx3_beamline_library.devices.detectors import md_camera  # noqa
from time import sleep
import matplotlib.pyplot as plt
from mx3_beamline_library.devices.motors import md3



# Instantiate run engine an start plan
RE = RunEngine({})
bec = BestEffortCallback()
RE.subscribe(bec)


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


drop_locations = ["B1-1"]
RE(get_drop_images(drop_locations, alignment_y_offset=0.25, alignment_z_offset=-1.0,))