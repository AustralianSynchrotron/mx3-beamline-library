from os import environ  # noqa

environ["DECTRIS_DETECTOR_HOST"] = "0.0.0.0"  # noqa
environ["DECTRIS_DETECTOR_PORT"] = "8000"  # noqa
environ["BL_ACTIVE"] = "True"  # noqa
environ["BLUESKY_DEBUG_CALLBACKS"] = "1"  # noqa
environ["SETTLE_TIME"] = "0.2"  # noqa

from typing import Generator

import lucid3
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.plan_stubs import mv, mvr
from bluesky.utils import Msg

from mx3_beamline_library.devices import detectors, motors
from mx3_beamline_library.devices.classes.detectors import BlackFlyCam
from mx3_beamline_library.devices.classes.motors import CosylabMotor
from mx3_beamline_library.plans.basic_scans import grid_scan


def save_image(data: npt.NDArray, screen_coordinates: list, filename: str):
    plt.figure()
    plt.imshow(data)
    plt.scatter(screen_coordinates[1], screen_coordinates[2], s=200, c="r", marker="+")
    plt.savefig(filename)
    plt.close()


def take_snapshot(camera: BlackFlyCam, filename: str, screen_coordinates: list = None):
    if screen_coordinates is None:
        screen_coordinates = [612, 512]
    plt.figure()
    array_data: npt.NDArray = camera.array_data.get()
    data = array_data.reshape(
        camera.height.get(), camera.width.get(), camera.depth.get()
    )
    plt.imshow(data)
    plt.scatter(screen_coordinates[0], screen_coordinates[1], s=200, c="r", marker="+")
    plt.savefig(filename)
    plt.close()


testrig = motors.testrig
motor_x = testrig.x
motor_x.wait_for_connection()
motor_z = testrig.z
motor_z.wait_for_connection()
motor_y = testrig.y
motor_y.wait_for_connection()
motor_phi = testrig.phi
motor_phi.wait_for_connection()

blackfly_camera = detectors.blackfly_camera
blackfly_camera.wait_for_connection()

dectris_detector = detectors.dectris_detector


def plot_raster_grid(
    camera: BlackFlyCam,
    initial_pos_pixels: list[int, int],
    final_pos_pixels: list[int, int],
    filename: str,
) -> None:
    """
    Plots the limits of the raster grid on top of the image taken from the
    camera.

    Parameters
    ----------
    camera : BlackFlyCam
        A blackfly camera
    initial_pos_pixels : list[int, int]
        The x and z coordinates of the initial position of the grid
    final_pos_pixels : list[int, int]
        The x and z coordinates of the final position of the grid
    filename : str
        The name of the JPEG file

    Returns
    -------
    None
    """
    plt.figure()
    array_data: npt.NDArray = camera.array_data.get()
    data = array_data.reshape(
        blackfly_camera.height.get(),
        blackfly_camera.width.get(),
        blackfly_camera.depth.get(),
    ).astype(np.uint8)

    screen_coordinates = lucid3.find_loop(
        image=data,
        rotation=True,
        rotation_k=1,
    )
    # Plot loop edge which is used as a reference to plot the grid
    plt.scatter(screen_coordinates[1], screen_coordinates[2], s=200, c="b", marker="+")

    plt.imshow(data)

    # Plot Grid limits:
    # Top
    x = np.linspace(initial_pos_pixels[0], final_pos_pixels[0], 100)
    z = initial_pos_pixels[1] * np.ones(len(x))
    plt.plot(x, z, color="red", linestyle="--")

    # Bottom
    x = np.linspace(initial_pos_pixels[0], final_pos_pixels[0], 100)
    z = final_pos_pixels[1] * np.ones(len(x))
    plt.plot(x, z, color="red", linestyle="--")

    # Right side
    z = np.linspace(initial_pos_pixels[1], final_pos_pixels[1], 100)
    x = final_pos_pixels[0] * np.ones(len(x))
    plt.plot(x, z, color="red", linestyle="--")

    # Left side
    z = np.linspace(initial_pos_pixels[1], final_pos_pixels[1], 100)
    x = initial_pos_pixels[0] * np.ones(len(x))
    plt.plot(x, z, color="red", linestyle="--")

    plt.savefig(filename)
    plt.close()


def move_motors_to_loop_edge(
    motor_x: CosylabMotor, motor_z: CosylabMotor, camera: BlackFlyCam
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

    Yields
    ------
    Generator[Msg, None, None]
        A message that tells bluesky to move the motors to the edge of the loop
    """
    beam_position = [612, 512]
    pixels_per_mm = [292.87, 292.87]

    array_data: npt.NDArray = camera.array_data.get()
    data = array_data.reshape(
        camera.height.get(), camera.width.get(), camera.depth.get()
    )

    screen_coordinates = lucid3.find_loop(
        image=data,
        rotation=True,
        rotation_k=1,
    )
    save_image(
        data,
        screen_coordinates,
        f"figs/step_2_loop_centering_fig_{screen_coordinates[1]}",
    )

    loop_position_x = (
        motor_x.position + (screen_coordinates[1] - beam_position[0]) / pixels_per_mm[0]
    )
    loop_position_z = (
        motor_z.position + (screen_coordinates[2] - beam_position[1]) / pixels_per_mm[1]
    )
    print("loop_position_x", loop_position_x)
    print("loop_position_z", loop_position_z)
    yield from mv(motor_x, loop_position_x)
    yield from mv(motor_z, loop_position_z)


def optical_centering(
    motor_x: CosylabMotor,
    motor_z: CosylabMotor,
    motor_phi: CosylabMotor,
    camera: BlackFlyCam,
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

    Yields
    ------
    Generator[Msg, None, None]
        A plan that automatically centers a loop
    """

    yield from move_motors_to_loop_edge(motor_x, motor_z, camera)
    yield from mvr(motor_phi, 90)
    yield from move_motors_to_loop_edge(motor_x, motor_z, camera)
    yield from mvr(motor_phi, 90)
    yield from move_motors_to_loop_edge(motor_x, motor_z, camera)

    loop_size = 0.6
    yield from mvr(motor_z, -loop_size)

    take_snapshot(camera, "figs/step_2_centered_loop")


def prepare_raster_grid(
    camera: BlackFlyCam,
    motor_x: CosylabMotor,
    motor_z: CosylabMotor = None,
    horizontal_scan=False,
) -> dict:
    """
    Prepares a raster grid

    Parameters
    ----------
    camera : BlackFlyCam
        Camera
    motor_x : CosylabMotor
        Motor X
    motor_z : CosylabMotor, optional
        Motor Z, by default None
    horizontal_scan : bool, optional
        If True, we prepare a hotizantal grid. By default False

    Returns
    -------
    dict
        A dictionary containing the initial and final positions of the grid.
    """
    beam_position = [612, 512]
    pixels_per_mm = [292.87, 292.87]
    loop_size = 0.6

    array_data: npt.NDArray = camera.array_data.get()
    data = array_data.reshape(
        camera.height.get(), camera.width.get(), camera.depth.get()
    )

    screen_coordinates = lucid3.find_loop(
        image=data,
        rotation=True,
        rotation_k=1,
    )

    if horizontal_scan:
        delta_z = 0
        delta_x = loop_size * pixels_per_mm[1]

        initial_pos_pixels = [beam_position[0] - delta_x, beam_position[1]]
        final_pos_pixels = [beam_position[0] + delta_x, beam_position[1]]
        plot_raster_grid(
            camera, initial_pos_pixels, final_pos_pixels, "figs/step_7_horizontal_scan"
        )

        initial_pos_z = None
        final_pos_z = None
    else:
        delta_z = abs(screen_coordinates[2] - beam_position[1])
        delta_x = delta_z

        initial_pos_pixels = [beam_position[0] - delta_z, beam_position[1] - delta_z]
        final_pos_pixels = [beam_position[0] + delta_z, beam_position[1] + delta_z]
        plot_raster_grid(
            camera, initial_pos_pixels, final_pos_pixels, "figs/step_3_prep_raster"
        )

        initial_pos_z = motor_z.position - delta_z / pixels_per_mm[1]
        final_pos_z = motor_z.position + delta_z / pixels_per_mm[1]

    initial_pos_x = motor_x.position - delta_x / pixels_per_mm[1]
    final_pos_x = motor_x.position + delta_x / pixels_per_mm[1]

    return {
        "initial_pos_x": initial_pos_x,
        "final_pos_x": final_pos_x,
        "initial_pos_z": initial_pos_z,
        "final_pos_z": final_pos_z,
    }


def master_plan(
    motor_x: CosylabMotor,
    motor_z: CosylabMotor,
    motor_phi: CosylabMotor,
    camera: BlackFlyCam,
) -> Generator[Msg, None, None]:
    """
    A bluesky plan that centers a sample following the procedure defined in Fig. 2
    of Hirata et al. (2019). Acta Cryst. D75, 138-150.

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

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan tha centers the a sample using optical and X-ray centering
    """

    # Step 2: Loop centering
    yield from optical_centering(motor_x, motor_z, motor_phi, camera)

    # Step 3: Prepare raster grid
    grid = prepare_raster_grid(camera, motor_x, motor_z)

    # Step 4: Raster scan
    yield from grid_scan(
        [dectris_detector],
        motor_z,
        grid["initial_pos_z"],
        grid["final_pos_z"],
        2,
        motor_x,
        grid["initial_pos_x"],
        grid["final_pos_x"],
        2,
        md={"sample_id": "test"},
    )

    # Steps 5 and 6: Find crystal and 2D centering
    # These values should come from the mx-spotfinder, but lets hardcode them for now
    yield from mv(motor_x, 0)
    yield from mv(motor_z, 0)

    # Step 7: Vertical scan
    yield from mvr(motor_phi, 90)
    horizontal_grid = prepare_raster_grid(camera, motor_x, horizontal_scan=True)
    yield from grid_scan(
        [dectris_detector],
        motor_x,
        horizontal_grid["initial_pos_x"],
        horizontal_grid["final_pos_x"],
        2,
        md={"sample_id": "test"},
    )


bec = BestEffortCallback()
RE = RunEngine({})
RE.subscribe(bec)

# We will need a low-resolution pin centering to get the loop in the camera view,
# but for now we move the motors to a position where we can see the loop
RE(mv(motor_x, 0, motor_z, 0, motor_phi, 0))
# RE(mv(motor_z, 0))
# RE(mv(motor_phi, 0))


print("starting loop centering")

RE(master_plan(motor_x, motor_z, motor_phi, blackfly_camera))
