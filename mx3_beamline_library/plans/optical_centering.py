import ast
import logging
import pickle
from os import environ
from typing import Generator, Optional, Union

import lucid3
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import redis
from bluesky.plan_stubs import mv, mvr
from bluesky.utils import Msg
from pydantic import BaseModel

from mx3_beamline_library.devices.classes.detectors import BlackFlyCam, DectrisDetector
from mx3_beamline_library.devices.classes.motors import CosylabMotor
from mx3_beamline_library.plans.basic_scans import grid_scan
import cv2


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
    data: npt.NDArray, screen_coordinates: list[int, int, int], filename: str
) -> None:
    """
    Saves an image from a numpy array taken from the camera ophyd object,
    and draws a red cross at the screen_coordinates.

    Parameters
    ----------
    data : npt.NDArray
        A numpy array containing an image from the camera
    screen_coordinates : list
        A list containing lucid3 results
    filename : str
        The filename

    Returns
    -------
    None
    """
    plt.figure()
    plt.imshow(data)
    plt.scatter(screen_coordinates[1], screen_coordinates[2], s=200, c="r", marker="+")
    plt.savefig(filename)
    plt.close()

def calculate_laplacian_variance(camera: BlackFlyCam) -> float:
    array_data: npt.NDArray = camera.array_data.get()
    data = array_data.reshape(
        camera.height.get(),
        camera.width.get(),
        camera.depth.get(),
    ).astype(np.uint8)

    screen_coordinates = lucid3.find_loop(
        image=data,
        rotation=True,
        rotation_k=1,
    )
    logging.getLogger("bluesky").info(screen_coordinates)

    if screen_coordinates[0] == "No loop detected":
        return None

    gray_image = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)

    return cv2.Laplacian(gray_image, cv2.CV_64F).var()

def unblur_image(camera: BlackFlyCam, motor_y: CosylabMotor):
    laplacian_variance = []
    motor_position = []
    laplacian_variance.append(calculate_laplacian_variance(camera))
    motor_position.append(motor_y.position)

    step_size = 0.1
    tolerance = 4

    yield from mvr(motor_y, step_size)
    laplacian_variance.append(calculate_laplacian_variance(camera))
    motor_position.append(motor_y.position)
    logging.info(f"motor position: {motor_position}")
    diff = laplacian_variance[0] - laplacian_variance[-1]
    if diff > 0:
        logging.getLogger("bluesky").info("Changing motor direction")
        step_size *=-1
        yield from mvr(motor_y, 2*step_size)
        laplacian_variance.append(calculate_laplacian_variance(camera))
        motor_position.append(motor_y.position)
        logging.info(f"motor position: {motor_position}")

    count = 0
    while abs(diff)<tolerance:
        yield from mvr(motor_y, step_size)
        laplacian_variance.append(calculate_laplacian_variance(camera))
        motor_position.append(motor_y.position)
        logging.info(f"motor position: {motor_position}")
        diff = laplacian_variance[0] - laplacian_variance[-1]
        count +=1
        if count >5:
            logging.getLogger("bluesky").info("couldn't find a best candidate, moving to the best position")
            logging.getLogger("bluesky").info(f"after {len(laplacian_variance)} iterations")
            break

    best_position = motor_position[np.argmax(laplacian_variance)]
    logging.getLogger("bluesky").info(f"best_position: {best_position}")
    logging.getLogger("bluesky").info(f"laplacian variance: {laplacian_variance}, {len(laplacian_variance)}")
    logging.getLogger("bluesky").info(f"motor_positions, {motor_position}, {len(motor_position)}")
    yield from mv(motor_y, best_position)


def move_motors_to_loop_edge(
    motor_x: CosylabMotor,
    motor_z: CosylabMotor,
    camera: BlackFlyCam,
    plot: bool = False,
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

    Yields
    ------
    Generator[Msg, None, None]
        A message that tells bluesky to move the motors to the edge of the loop
    """
    array_data: npt.NDArray = camera.array_data.get()
    data = array_data.reshape(
        camera.height.get(), camera.width.get(), camera.depth.get()
    )

    screen_coordinates = lucid3.find_loop(
        image=data,
        rotation=True,
        rotation_k=1,
    )

    if plot:
        save_image(
            data,
            screen_coordinates,
            f"step_2_loop_centering_fig_{screen_coordinates[1]}",
        )

    loop_position_x = (
        motor_x.position + (screen_coordinates[1] - BEAM_POSITION[0]) / PIXELS_PER_MM_X
    )
    loop_position_z = (
        motor_z.position + (screen_coordinates[2] - BEAM_POSITION[1]) / PIXELS_PER_MM_Z
    )
    yield from mv(motor_x, loop_position_x, motor_z, loop_position_z)

def optical_centering(
    motor_x: CosylabMotor,
    motor_y: CosylabMotor,
    motor_z: CosylabMotor,
    motor_phi: CosylabMotor,
    camera: BlackFlyCam,
    plot: bool = False,
) -> Generator[Msg, None, None]:
    """
    Automatically centers the loop using Lucid3, following the method outlined
    in Fig. 5 of Hirata et al. (2019). Acta Cryst. D75, 138-150.


    Parameters
    ----------
    motor_x : CosylabMotor
        Motor X
    motor_z : CosylabMotor
        Motor Z
    motor_phi : CosylabMotor
        Motor Phi
    camera : BlackFlyCam
        Camera
    plot : bool
        If true, we take snapshot of the centered loop, by default False

    Yields
    ------
    Generator[Msg, None, None]
        A plan that automatically centers a loop
    """
    yield from mv(motor_phi, 0)
    logging.info(f"------------------Omega----------: {motor_phi.position}")
    yield from unblur_image(camera, motor_y)
    yield from move_motors_to_loop_edge(motor_x, motor_z, camera, plot)
    yield from mv(motor_phi, 90)
    logging.info(f"------------------Omega----------: {motor_phi.position}")
    yield from unblur_image(camera, motor_y)
    yield from move_motors_to_loop_edge(motor_x, motor_z, camera, plot)
    yield from mv(motor_phi, 180)
    logging.info(f"------------------Omega----------: {motor_phi.position}")
    yield from unblur_image(camera, motor_y)
    yield from move_motors_to_loop_edge(motor_x, motor_z, camera, plot)

    loop_size = 0.6
    yield from mvr(motor_z, -loop_size)

    if plot:
        take_snapshot(camera, "step_2_centered_loop")
