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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

REDIS_HOST = environ.get("REDIS_HOST", "0.0.0.0")
REDIS_PORT = int(environ.get("REDIS_PORT", "6379"))

redis_connection = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)


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
        The name of the PNG file

    Returns
    -------
    None
    """
    plt.figure()
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
    # Plot loop edge which is used as a reference to plot the grid
    plt.scatter(screen_coordinates[1], screen_coordinates[2], s=200, c="b", marker="+")

    plt.imshow(data)

    # Plot grid:
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

    if plot:
        save_image(
            data,
            screen_coordinates,
            f"step_2_loop_centering_fig_{screen_coordinates[1]}",
        )

    loop_position_x = (
        motor_x.position + (screen_coordinates[1] - beam_position[0]) / pixels_per_mm[0]
    )
    loop_position_z = (
        motor_z.position + (screen_coordinates[2] - beam_position[1]) / pixels_per_mm[1]
    )
    yield from mv(motor_x, loop_position_x)
    yield from mv(motor_z, loop_position_z)


def optical_centering(
    motor_x: CosylabMotor,
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

    yield from move_motors_to_loop_edge(motor_x, motor_z, camera, plot)
    yield from mvr(motor_phi, 90)
    yield from move_motors_to_loop_edge(motor_x, motor_z, camera, plot)
    yield from mvr(motor_phi, 90)
    yield from move_motors_to_loop_edge(motor_x, motor_z, camera, plot)

    loop_size = 0.6
    yield from mvr(motor_z, -loop_size)

    if plot:
        take_snapshot(camera, "step_2_centered_loop")


def prepare_raster_grid(
    camera: BlackFlyCam,
    motor_x: CosylabMotor,
    motor_z: CosylabMotor = None,
    horizontal_scan: bool = False,
    plot: bool = False,
) -> dict:
    """
    Prepares a raster grid. If horizontal_scan=False, we create a square
    grid of length=2*loop_radius, otherwise we create a horizontal grid of
    length=2*loop_radius

    Parameters
    ----------
    camera : BlackFlyCam
        Camera
    motor_x : CosylabMotor
        Motor X
    motor_z : CosylabMotor, optional
        Motor Z, by default None
    horizontal_scan : bool, optional
        If True, we prepare a horizontal grid. By default False
    plot : bool, optional
        If True, we plot the raster grid and save it to a file, by default False

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

        if plot:
            plot_raster_grid(
                camera, initial_pos_pixels, final_pos_pixels, "step_7_horizontal_scan"
            )

        initial_pos_z = None
        final_pos_z = None
    else:
        delta_z = abs(screen_coordinates[2] - beam_position[1])
        delta_x = delta_z

        initial_pos_pixels = [beam_position[0] - delta_z, beam_position[1] - delta_z]
        final_pos_pixels = [beam_position[0] + delta_z, beam_position[1] + delta_z]

        if plot:
            plot_raster_grid(
                camera, initial_pos_pixels, final_pos_pixels, "step_3_prep_raster"
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


class SpotfinderResults(BaseModel):
    type: str
    number_of_spots: int
    image_id: int
    sequence_id: int
    bluesky_event_doc: Union[bytes, dict]


class BlueskyEventData(BaseModel):
    dectris_detector_sequence_id: int
    testrig_x_user_setpoint: Optional[float]
    testrig_x: Optional[float]
    testrig_z_user_setpoint: Optional[float]
    testrig_z: Optional[float]


class SpotfinderAndBlueskyMetadata(BaseModel):
    spotfinder_results: SpotfinderResults
    bluesky_event_data: BlueskyEventData


def read_message_from_redis_streams(
    topic: str, id: int
) -> tuple[SpotfinderAndBlueskyMetadata, Union[int, bytes]]:
    """
    Reads pickled messages from a redis stream

    Parameters
    ----------
    topic : str
        Name of the topic of the redis stream, aka, the sample_id
    id : Union[bytes, int]
        id of the topic in bytes or int format

    Returns
    -------
    bluesky_event_doc : tuple[dict, bytes]
        A tuple containing a dictionary and the last id.
        The diction ary has the following keys:
        b'type', b'number_of_spots', b'image_id', and b'sequence_id'
    """
    response = redis_connection.xread({topic: id}, count=1)

    # Extract key and messages from the response
    _, messages = response[0]

    # Update last_id and store messages data
    last_id, data = messages[0]

    spotfinder_results = SpotfinderResults(
        type=data[b"type"],
        number_of_spots=data[b"number_of_spots"],
        image_id=data[b"image_id"],
        sequence_id=data[b"sequence_id"],
        bluesky_event_doc=pickle.loads(data[b"bluesky_event_doc"]),
    )
    bluesky_event_data = BlueskyEventData.parse_obj(
        spotfinder_results.bluesky_event_doc["data"]
    )

    sequence_id_zmq = spotfinder_results.sequence_id
    sequence_id_bluesky_doc = bluesky_event_data.dectris_detector_sequence_id

    assert sequence_id_zmq == sequence_id_bluesky_doc, (
        "Sequence_id obtained from bluesky doc is different from the ZMQ sequence_id "
        f"sequence_id_zmq: {sequence_id_zmq}, "
        f"sequence_id_bluesky_doc: {sequence_id_bluesky_doc}"
    )

    spotfinder_and_bluesky_metadata = SpotfinderAndBlueskyMetadata(
        spotfinder_results=spotfinder_results,
        bluesky_event_data=bluesky_event_data,
        last_id=last_id,
    )

    return spotfinder_and_bluesky_metadata, last_id


def find_crystal_position(
    sample_id: str, last_id: Union[int, bytes]
) -> tuple[SpotfinderAndBlueskyMetadata, Union[int, bytes]]:

    result = []
    number_of_spots_list = []
    for _ in range(redis_connection.xlen(sample_id)):

        spotfinder_and_bluesky_metadata, last_id = read_message_from_redis_streams(
            sample_id, last_id
        )
        result.append(spotfinder_and_bluesky_metadata)
        number_of_spots_list.append(
            spotfinder_and_bluesky_metadata.spotfinder_results.number_of_spots
        )

    argmax = np.argmax(number_of_spots_list)
    print("Max number of spots:", number_of_spots_list[argmax])
    print(result[argmax])
    return result[argmax], last_id


def optical_and_xray_centering(
    detector: DectrisDetector,
    motor_x: CosylabMotor,
    motor_z: CosylabMotor,
    motor_phi: CosylabMotor,
    camera: BlackFlyCam,
    md: dict,
    plot: bool = False,
) -> Generator[Msg, None, None]:
    """
    A bluesky plan that centers a sample following the procedure defined in Fig. 2
    of Hirata et al. (2019). Acta Cryst. D75, 138-150.

    Parameters
    ----------
    detector: DectrisDetector
        The dectris detector ophyd device
    motor_x : CosylabMotor
        Motor X
    motor_z : CosylabMotor
        Motor Z
    motor_phi : CosylabMotor
        Motor Phi
    camera : BlackFlyCam
        Camera
    md : dict
        Bluesky metadata, generally we include here the sample id,
        e.g. {"sample_id": "test_sample"}
    plot : bool
        If true, we take snapshots of the plan at different stages for debugging purposes.
        By default false

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan tha centers the a sample using optical and X-ray centering
    """

    # Step 2: Loop centering
    logging.info("Step 2: Loop centering")
    yield from optical_centering(motor_x, motor_z, motor_phi, camera, plot)

    # Step 3: Prepare raster grid
    logging.info("Step 3: Prepare raster grid")
    grid = prepare_raster_grid(
        camera, motor_x, motor_z, horizontal_scan=False, plot=plot
    )
    # Step 4: Raster scan
    logging.info("Step 4: Raster scan")
    yield from grid_scan(
        [detector],
        motor_z,
        grid["initial_pos_z"],
        grid["final_pos_z"],
        2,
        motor_x,
        grid["initial_pos_x"],
        grid["final_pos_x"],
        2,
        md=md,
    )

    # Steps 5 and 6: Find crystal and 2D centering
    # These values should come from the mx-spotfinder, but lets hardcode them for now
    logging.info("Steps 5 and 6: Find crystal and 2D centering")
    crystal_position, last_id = find_crystal_position(md["sample_id"], last_id=0)
    logging.info(
        "Max number of spots:", crystal_position.spotfinder_results.number_of_spots
    )
    logging.info("Motor X position:", crystal_position.bluesky_event_data.testrig_x)
    logging.info("Motor Z position:", crystal_position.bluesky_event_data.testrig_z)
    yield from mv(
        motor_x,
        crystal_position.bluesky_event_data.testrig_x,
        motor_z,
        crystal_position.bluesky_event_data.testrig_z,
    )

    # Step 7: Vertical scan
    logging.info("Step 7: Vertical scan")
    yield from mvr(motor_phi, 90)
    horizontal_grid = prepare_raster_grid(
        camera, motor_x, horizontal_scan=True, plot=plot
    )
    yield from grid_scan(
        [detector],
        motor_x,
        horizontal_grid["initial_pos_x"],
        horizontal_grid["final_pos_x"],
        2,
        md=md,
    )
    crystal_position, last_id = find_crystal_position(md["sample_id"], last_id=last_id)
    logging.info(
        "Max number of spots:", crystal_position.spotfinder_results.number_of_spots
    )
    logging.info("Motor X position:", crystal_position.bluesky_event_data.testrig_x)
    logging.info("Motor Z position:", crystal_position.bluesky_event_data.testrig_z)
    yield from mv(
        motor_x,
        crystal_position.bluesky_event_data.testrig_x,
    )
