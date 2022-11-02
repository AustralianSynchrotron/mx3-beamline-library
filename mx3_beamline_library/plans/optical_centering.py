import ast
import logging
from os import environ
from typing import Generator

import cv2
import lucid3
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from bluesky.plan_stubs import mv
from bluesky.utils import Msg

from mx3_beamline_library.devices.classes.detectors import BlackFlyCam
from mx3_beamline_library.devices.classes.motors import CosylabMotor
from mx3_beamline_library.plans.psi_optical_centering import loopImageProcessing

BEAM_POSITION = ast.literal_eval(environ.get("BEAM_POSITION", "[612, 512]"))
PIXELS_PER_MM_X = float(environ.get("PIXELS_PER_MM_X", "292.87"))
PIXELS_PER_MM_Z = float(environ.get("PIXELS_PER_MM_Z", "292.87"))


def take_snapshot(
    camera: BlackFlyCam, filename: str, screen_coordinates: tuple[int, int] = (612, 512)
) -> None:
    """
    Saves an image given the ophyd camera object,
    and draws a red cross at the screen_coordinates.


    Parameters
    ----------
    camera : BlackFlyCam
        A blackfly camera ophyd device
    filename : str
        The filename
    screen_coordinates : tuple[int, int], optional
        The screen coordinates, by default (612, 512)

    Returns
    -------
    None
    """
    plt.figure()
    array_data: npt.NDArray = camera.array_data.get()
    data = array_data.reshape(
        camera.height.get(), camera.width.get(), camera.depth.get()
    )
    plt.imshow(data)
    plt.scatter(screen_coordinates[0], screen_coordinates[1], s=200, c="r", marker="+")
    plt.savefig(filename)
    plt.close()


def save_image(
    data: npt.NDArray, x_coord: float, y_coord: float, filename: str
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
    plt.figure()
    plt.imshow(data)
    plt.scatter(x_coord, y_coord, s=200, c="r", marker="+")
    plt.savefig(filename)
    plt.close()


def calculate_variance(camera: BlackFlyCam) -> float:
    """
    We calculate the variance of the convolution of the laplacian kernel with an image,
    e.g. var( Img * L(x,y) ), where Img is an image taken from the camera ophyd object,
    and L(x,y) is the Laplacian kernel.

    Parameters
    ----------
    camera : BlackFlyCam
        A camera ophyd object

    Returns
    -------
    float
        var( Img * L(x,y) )
    """
    array_data: npt.NDArray = camera.array_data.get()
    data = array_data.reshape(
        camera.height.get(),
        camera.width.get(),
        camera.depth.get(),
    ).astype(np.uint16)

    gray_image = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)

    return cv2.Laplacian(gray_image, cv2.CV_64F).var()


def unblur_image(
    camera: BlackFlyCam,
    focus_motor: CosylabMotor,
    a: float = 0.0,
    b: float = 1.0,
    tol: float = 0.2,
) -> float:
    """
    We use the Golden-section search to find the maximum of the variance function described in
    the calculate_variance method ( `var( Img * L(x,y)` ) ). We assume that the function
    is strictly unimodal on [a,b].
    See for example: https://en.wikipedia.org/wiki/Golden-section_search

    Parameters
    ----------
    camera : BlackFlyCam
        A camera ophyd object
    focus_motor : CosylabMotor
        Motor used for focusing the camera
    a : float
        Minimum value to search for the maximum of var( Img * L(x,y) )
    b : float
        Maximum value to search for the maximum of var( Img * L(x,y) )
    tol : float, optional
        The tolerance, by default 0.2

    Returns
    -------
    Generator[Msg, None, None]
        Moves motor_y to a position where the image is focused
    """
    gr = (np.sqrt(5) + 1) / 2

    c = b - (b - a) / gr
    d = a + (b - a) / gr

    count = 0
    logging.getLogger("bluesky.RE.msg").info("Focusing image...")
    while abs(b - a) > tol:
        yield from mv(focus_motor, c)
        val_c = calculate_variance(camera)

        yield from mv(focus_motor, d)
        val_d = calculate_variance(camera)

        if val_c > val_d:  # val_c > val_d to find the maximum
            b = d
        else:
            a = c

        # We recompute both c and d here to avoid loss of precision which
        # may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        count += 1
        logging.getLogger("bluesky.RE.msg").info(f"Iteration: {count}")
    maximum = (b + a) / 2
    logging.getLogger("bluesky.RE.msg").info(f"Optimal motor_y value: {maximum}")
    yield from mv(focus_motor, maximum)


def move_motors_to_loop_edge(
    motor_x: CosylabMotor,
    motor_z: CosylabMotor,
    camera: BlackFlyCam,
    plot: bool = False,
    method: str = "psi",
) -> Generator[Msg, None, None]:
    """
    Moves the motor_x and motor_z to the edge of the loop. The edge of the loop is found
    using Lucid3

    Parameters
    ----------
    motor_x : CosylabMotor
        Motor x
    motor_z : CosylabMotor
        Motor z
    camera : BlackFlyCam
        Camera
    plot : bool
        If true, we take snapshot of edge of the loop and save it to a file, by default False
    method : str
        Method used to find the edge of the loop. Could be either
        lucid3 or psi.

    Raises
    ------
    NotImplementedError
        An error if method is not lucid3 or psi

    Yields
    ------
    Generator[Msg, None, None]
        A message that tells bluesky to move the motors to the edge of the loop
    """
    array_data: npt.NDArray = camera.array_data.get()
    data = array_data.reshape(
        camera.height.get(), camera.width.get(), camera.depth.get()
    ).astype(np.uint8)
    if method.lower() == "lucid3":
        loop_detected, x_coord, y_coord = lucid3.find_loop(
            image=data,
            rotation=True,
            rotation_k=2,
        )
    elif method.lower() == "psi":
        procImg = loopImageProcessing(data)
        procImg.findContour(zoom="-208.0", beamline="X06DA")
        extremes = procImg.findExtremes()
        screen_coordinates = extremes["bottom"]
        x_coord = screen_coordinates[0]
        y_coord = screen_coordinates[1]
    else:
        raise NotImplementedError(f"Supported methods are lucid3 or psi, not {method}")

    logging.getLogger("bluesky").info(f"screen coordinates: {screen_coordinates}")

    if plot:
        save_image(
            data,
            x_coord,
            y_coord,
            f"step_2_loop_centering_fig_{x_coord}",
        )

    loop_position_x = motor_x.position + (x_coord - BEAM_POSITION[0]) / PIXELS_PER_MM_X
    loop_position_z = motor_z.position + (y_coord - BEAM_POSITION[1]) / PIXELS_PER_MM_Z
    yield from mv(motor_x, loop_position_x, motor_z, loop_position_z)


def optical_centering(
    motor_x: CosylabMotor,
    motor_y: CosylabMotor,
    motor_z: CosylabMotor,
    motor_phi: CosylabMotor,
    camera: BlackFlyCam,
    plot: bool = False,
    auto_focus: bool = True,
    min_focus: float = 0.0,
    max_focus: float = 1.0,
    tol: float = 0.3,
    method: str = "psi",
) -> Generator[Msg, None, None]:
    """
    Automatically centers the loop using Lucid3. Before analysing an image
    with Lucid3, we unblur the image to make sure the Lucid3 results are consistent

    Parameters
    ----------
    motor_x : CosylabMotor
        Motor X
    motor_y : CosylabMotor
        Motor Y
    motor_z : CosylabMotor
        Motor Z
    motor_phi : CosylabMotor
        Omega
    camera : BlackFlyCam
        Camera
    plot : bool, optional
        If true, we take snapshot of the centered loop, by default False
    auto_focus : bool, optional
        If true, we autofocus the image before analysing an image with Lucid3,
        by default True
    min_focus : float, optional
        Minimum value to search for the maximum of var( Img * L(x,y) ),
        by default 0.0
    max_focus : float, optional
        Maximum value to search for the maximum of var( Img * L(x,y) ),
        by default 1.0
    tol : float, optional
        The tolerance used by the Golden-section search, by default 0.3

    Yields
    ------
    Generator[Msg, None, None]
        A plan that automatically centers a loop
    """
    yield from mv(motor_phi, 0)
    if auto_focus:
        yield from unblur_image(camera, motor_y, min_focus, max_focus, tol)
    logging.info(f"------------------Omega----------: {motor_phi.position}")
    yield from move_motors_to_loop_edge(motor_x, motor_z, camera, plot, method)

    yield from mv(motor_phi, 90)
    logging.info(f"------------------Omega----------: {motor_phi.position}")
    yield from move_motors_to_loop_edge(motor_x, motor_z, camera, plot, method)

    yield from mv(motor_phi, 180)
    logging.info(f"------------------Omega----------: {motor_phi.position}")
    yield from move_motors_to_loop_edge(motor_x, motor_z, camera, plot, method)

    if plot:
        take_snapshot(camera, "step_2_centered_loop")
