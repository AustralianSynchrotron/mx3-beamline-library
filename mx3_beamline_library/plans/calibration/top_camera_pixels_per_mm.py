import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from mx3_beamline_library.config import redis_connection
from mx3_beamline_library.devices import detectors
from mx3_beamline_library.devices.motors import md3
from mx3_beamline_library.logger import setup_logger
from mx3_beamline_library.plans.plan_stubs import md3_move
from mx3_beamline_library.schemas.optical_centering import OpticalCenteringExtraConfig
from mx3_beamline_library.science.optical_and_loop_centering.loop_edge_detection import (
    LoopEdgeDetection,
)

logger = setup_logger()

config = OpticalCenteringExtraConfig()
roi_x = config.top_camera.roi_x
roi_y = config.top_camera.roi_y


def _find_tip_of_loop(camera, plot=False):
    img = (
        camera.array_data.get()
        .reshape(camera.height.get(), camera.width.get())
        .astype(np.uint8)
    )
    img = img[roi_y[0] : roi_y[1], roi_x[0] : roi_x[1]]

    procImg = LoopEdgeDetection(img, block_size=49, adaptive_constant=6)
    screen_coordinates = procImg.find_tip()
    if plot:
        plt.figure()
        # plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.imshow(img)
        plt.savefig("raw_data")
        plt.close()
        _save_image(img, screen_coordinates, filename="top_camera")

    return screen_coordinates


def _save_image(
    data: npt.NDArray, screen_coordinates: npt.NDArray, filename: str
) -> None:
    """
    Saves an image from a numpy array taken from the camera ophyd object,
    and draws a red cross at the screen_coordinates.

    Parameters
    ----------
    data : npt.NDArray
        A numpy array containing an image from the camera
    x_coord : float
        X coordinate
    y_coord : float
        Y coordinate
    filename : str
        The filename

    Returns
    -------
    None
    """
    x_coord = screen_coordinates[0]
    y_coord = screen_coordinates[1]
    plt.figure()
    plt.imshow(data, cmap="gray", vmin=0, vmax=255)
    plt.scatter(
        x_coord,
        y_coord,
        s=200,
        c="r",
        marker="+",
    )
    # plt.legend()
    plt.savefig(filename)
    plt.close()


def _get_x_and_y_coords():
    camera = detectors.blackfly_camera
    camera.wait_for_connection()

    x_vals = []
    y_vals = []
    plt.figure()
    for _ in range(30):
        coords = _find_tip_of_loop(camera)
        x_vals.append(coords[0])
        y_vals.append(coords[1])
        plt.scatter(
            coords[0],
            coords[1],
            s=200,
            c="r",
            marker="+",
        )
    data = (
        camera.array_data.get()
        .reshape(camera.height.get(), camera.width.get())
        .astype(np.uint8)
    )
    data = data[roi_y[0] : roi_y[1], roi_x[0] : roi_x[1]]

    y_median = np.median(y_vals)
    # print("x_median", y_median)
    np.std(y_vals)
    # print("std", y_std)

    x_median = np.median(x_vals)
    # print("y_median", x_median)
    np.std(x_vals)
    # print("std", x_std)
    return x_median, y_median


def _get_pixels_per_mm_y():
    start_alignment_y = 0
    start_sample_x = 0
    start_sample_y = 0
    start_omega = 0
    start_alignment_z = 0
    yield from md3_move(
        md3.omega,
        start_omega,
        md3.alignment_y,
        start_alignment_y,
        md3.sample_x,
        start_sample_x,
        md3.sample_y,
        start_sample_y,
        md3.alignment_z,
        start_alignment_z,
    )
    start_pixel_x, start_pixel_y = _get_x_and_y_coords()
    print("start pos done")

    end_alignment_y = 1
    yield from md3_move(md3.alignment_y, end_alignment_y)
    end_pixel_x, end_pixel_y = _get_x_and_y_coords()
    print("end pos done")

    pixels_per_mm_y = abs(start_pixel_y - end_pixel_y) / abs(
        start_alignment_y - end_alignment_y
    )
    logger.info(f"Pixels per mm y: {pixels_per_mm_y}")
    return pixels_per_mm_y


def _get_pixels_per_mm_x():
    start_alignment_y = 0
    start_sample_x = 0
    start_sample_y = 0
    start_omega = 0
    start_alignment_z = 0
    yield from md3_move(
        md3.omega,
        start_omega,
        md3.alignment_y,
        start_alignment_y,
        md3.sample_x,
        start_sample_x,
        md3.sample_y,
        start_sample_y,
        md3.alignment_z,
        start_alignment_z,
    )
    start_pixel_x, start_pixel_y = _get_x_and_y_coords()
    print("start pos done")

    end_alignment_z = 1
    yield from md3_move(md3.alignment_z, end_alignment_z)
    end_pixel_x, end_pixel_y = _get_x_and_y_coords()
    print("end pos done")

    pixels_per_mm_x = abs(start_pixel_x - end_pixel_x) / abs(
        start_alignment_z - end_alignment_z
    )
    logger.info(f"Pixels per mm x: {pixels_per_mm_x}")
    return pixels_per_mm_x


def set_x_and_y_pixels_per_mm():
    pixels_per_mm_x = yield from _get_pixels_per_mm_x()
    pixels_per_mm_y = yield from _get_pixels_per_mm_y()
    redis_connection.hset(
        "top_camera_pixels_per_mm",
        mapping={
            "pixels_per_mm_x": float(pixels_per_mm_x),
            "pixels_per_mm_y": float(pixels_per_mm_y),
        },
    )
