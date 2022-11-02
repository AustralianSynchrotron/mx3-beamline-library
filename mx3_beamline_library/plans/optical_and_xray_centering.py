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
from mx3_beamline_library.plans.optical_centering import optical_centering
from mx3_beamline_library.plans.psi_optical_centering import loopImageProcessing

REDIS_HOST = environ.get("REDIS_HOST", "0.0.0.0")
REDIS_PORT = int(environ.get("REDIS_PORT", "6379"))

redis_connection = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)

BEAM_POSITION = ast.literal_eval(environ.get("BEAM_POSITION", "[640, 512]"))
PIXELS_PER_MM_X = float(environ.get("PIXELS_PER_MM_X", "292.87"))
PIXELS_PER_MM_Z = float(environ.get("PIXELS_PER_MM_Z", "292.87"))


class TestrigEventData(BaseModel):
    dectris_detector_sequence_id: Union[int, float]
    testrig_x_user_setpoint: float
    testrig_x: float
    testrig_z_user_setpoint: Optional[float]
    testrig_z: Optional[float]


class BlueskyEventDoc(BaseModel):
    descriptor: str
    time: float
    data: Union[TestrigEventData, dict]
    timestamps: Union[TestrigEventData, dict]
    seq_num: int
    uid: str
    filled: dict


class SpotfinderResults(BaseModel):
    type: str
    number_of_spots: int
    image_id: int
    sequence_id: int
    bluesky_event_doc: Union[BlueskyEventDoc, dict, bytes]


class RasterGridCoordinates(BaseModel):
    initial_pos_x: float
    final_pos_x: float
    initial_pos_z: Optional[float]
    final_pos_z: Optional[float]


def plot_raster_grid(
        camera: BlackFlyCam,
        rectangle_coordinates: dict,
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
    )
    plt.imshow(data)

    # Plot grid:
    # Top
    plt.scatter(
        rectangle_coordinates["top_left"][0],
        rectangle_coordinates["top_left"][1],
        s=200,
        c="b",
        marker="+",
    )
    plt.scatter(
        rectangle_coordinates["bottom_right"][0],
        rectangle_coordinates["bottom_right"][1],
        s=200,
        c="b",
        marker="+",
    )

    # top
    x = np.linspace(
        rectangle_coordinates["top_left"][0],
        rectangle_coordinates["bottom_right"][0],
        100,
    )
    z = rectangle_coordinates["top_left"][1] * np.ones(len(x))
    plt.plot(x, z, color="red", linestyle="--")

    # Bottom
    x = np.linspace(
        rectangle_coordinates["top_left"][0],
        rectangle_coordinates["bottom_right"][0],
        100,
    )
    z = rectangle_coordinates["bottom_right"][1] * np.ones(len(x))
    plt.plot(x, z, color="red", linestyle="--")

    # Right side
    z = np.linspace(
        rectangle_coordinates["top_left"][1],
        rectangle_coordinates["bottom_right"][1],
        100,
    )
    x = rectangle_coordinates["bottom_right"][0] * np.ones(len(x))
    plt.plot(x, z, color="red", linestyle="--")

    # Left side
    z = np.linspace(
        rectangle_coordinates["top_left"][1],
        rectangle_coordinates["bottom_right"][1],
        100,
    )
    x = rectangle_coordinates["top_left"][0] * np.ones(len(x))
    plt.plot(x, z, color="red", linestyle="--")

    plt.savefig(filename)
    plt.close()


def prepare_raster_grid(
    camera: BlackFlyCam,
    motor_x: CosylabMotor,
    motor_z: CosylabMotor = None,
    plot: bool = False,
) -> RasterGridCoordinates:
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
    RasterGridCoordinates
        A pydantinc model containing the initial and final positions of the grid.
    """

    array_data: npt.NDArray = camera.array_data.get()
    data = array_data.reshape(
        camera.height.get(), camera.width.get(), camera.depth.get()
    ).astype(
        np.uint8
    )  # the code only works with np.uint8 data types

    #screen_coordinates = lucid3.find_loop(
    #    image=data,
    #    rotation=True,
    #    rotation_k=1,
    #)
    procImg = loopImageProcessing(data)
    procImg.findContour(zoom="-208.0", beamline="X06DA")
    extremes = procImg.findExtremes()
    rectangle_coordinates = procImg.fitRectangle()

    if plot:
        plot_raster_grid(
        camera,
        rectangle_coordinates,
        "step_3_prep_raster",
        )

    # Z motor positions
    initial_pos_z_pixels = abs(rectangle_coordinates["top_left"][1] - BEAM_POSITION[1])
    final_pos_z_pixels = abs(rectangle_coordinates["bottom_right"][1] - BEAM_POSITION[1])

    initial_pos_z = motor_z.position - initial_pos_z_pixels / PIXELS_PER_MM_Z
    final_pos_z = motor_z.position + final_pos_z_pixels / PIXELS_PER_MM_Z

    # X motor positions
    initial_pos_x_pixels = abs(rectangle_coordinates["top_left"][0] - BEAM_POSITION[0])
    final_pos_x_pixels = abs(rectangle_coordinates["bottom_right"][0] - BEAM_POSITION[0])

    initial_pos_x = motor_x.position - initial_pos_x_pixels / PIXELS_PER_MM_X
    final_pos_x = motor_x.position + final_pos_x_pixels / PIXELS_PER_MM_X

    coordinates = RasterGridCoordinates(
        initial_pos_x=initial_pos_x,
        final_pos_x=final_pos_x,
        initial_pos_z=initial_pos_z,
        final_pos_z=final_pos_z,
    )
    logging.getLogger("bluesky").info(f"Raster grid coordinates [mm]: {coordinates}")

    return coordinates


def read_message_from_redis_streams(
    topic: str, id: Union[bytes, int]
) -> tuple[SpotfinderResults, bytes]:
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
    spotfinder_results, last_id : tuple[SpotfinderResults, bytes]
        A tuple containing SpotfinderResults and the redis streams
        last_id
    """
    response = redis_connection.xread({topic: id}, count=1)

    # Extract key and messages from the response
    _, messages = response[0]

    # Update last_id and store messages data
    last_id, data = messages[0]
    bluesky_event_doc = BlueskyEventDoc.parse_obj(
        pickle.loads(data[b"bluesky_event_doc"])
    )
    bluesky_event_doc.data = TestrigEventData.parse_obj(bluesky_event_doc.data)

    spotfinder_results = SpotfinderResults(
        type=data[b"type"],
        number_of_spots=data[b"number_of_spots"],
        image_id=data[b"image_id"],
        sequence_id=data[b"sequence_id"],
        bluesky_event_doc=bluesky_event_doc,
    )

    sequence_id_zmq = spotfinder_results.sequence_id
    sequence_id_bluesky_doc = (
        spotfinder_results.bluesky_event_doc.data.dectris_detector_sequence_id
    )

    assert sequence_id_zmq == sequence_id_bluesky_doc, (
        "Sequence_id obtained from bluesky doc is different from the ZMQ sequence_id "
        f"sequence_id_zmq: {sequence_id_zmq}, "
        f"sequence_id_bluesky_doc: {sequence_id_bluesky_doc}"
    )
    return spotfinder_results, last_id


def find_crystal_position(
    sample_id: str, last_id: Union[int, bytes]
) -> tuple[SpotfinderResults, bytes]:
    """
    Finds the crystal position based on the number of spots obtained from a
    grid_scan. Data is obtained from redis streams, which is generated by the
    mx-spotfinder.

    Parameters
    ----------
    sample_id : str
        Sample id
    last_id : Union[int, bytes]
        Redis streams last_id

    Returns
    -------
    spotfinder_results, last_id : tuple[SpotfinderResults, bytes]
        SpotfinderResults corresponding to the maximum number of spots,
        and the redis_streams last_id
    """
    result = []
    number_of_spots_list = []
    for _ in range(redis_connection.xlen(sample_id)):
        try:
            spotfinder_results, last_id = read_message_from_redis_streams(
                sample_id, last_id
            )
            result.append(spotfinder_results)
            number_of_spots_list.append(spotfinder_results.number_of_spots)
        except IndexError:
            pass

    argmax = np.argmax(number_of_spots_list)
    logging.getLogger("bluesky.RE.msg").info(
        f"Max number of spots: {number_of_spots_list[argmax]}"
    )
    logging.getLogger("bluesky.RE.msg").info(
        f"Metadata associated with the max number of spots: {result[argmax]}"
    )
    return result[argmax], last_id


def optical_and_xray_centering(
    detector: DectrisDetector,
    motor_x: CosylabMotor,
    numer_of_steps_x: int,
    motor_y: CosylabMotor,
    motor_z: CosylabMotor,
    number_of_steps_z: int,
    motor_phi: CosylabMotor,
    camera: BlackFlyCam,
    md: dict,
    plot: bool = False,
    auto_focus: bool = True,
    min_focus: float = 0.0,
    max_focus: float = 1.0,
    tol: float = 0.3,
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
    numer_of_steps_x : int
        Number of steps (X axis)
    motor_y : CosylabMotor
        Motor Y
    motor_z : CosylabMotor
        Motor Z
    numer_of_steps_z : int
        Number of steps (Z axis)
    motor_phi : CosylabMotor
        Motor Phi
    camera : BlackFlyCam
        Camera
    md : dict
        Bluesky metadata, generally we include here the sample id,
        e.g. {"sample_id": "test_sample"}
    plot : bool, optional
        If true, we take snapshots of the plan at different stages for debugging purposes.
        By default false
    auto_focus : bool, optional
        If true, we autofocus the image before analysing an image with Lucid3,
        by default True
    min_focus : float, optional
        Minimum value to search for the maximum of var( Img * L(x,y) ),
        by default 0
    max_focus : float, optional
        Maximum value to search for the maximum of var( Img * L(x,y) ),
        by default 1
    tol : float, optional
        The tolerance used by the Golden-section search, by default 0.3

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan tha centers the a sample using optical and X-ray centering
    """

    # Step 2: Loop centering
    logging.getLogger("bluesky.RE.msg").info("Step 2: Loop centering")
    yield from optical_centering(
        motor_x,
        motor_y,
        motor_z,
        motor_phi,
        camera,
        plot,
        auto_focus,
        min_focus,
        max_focus,
        tol,
    )

    # Step 3: Prepare raster grid
    logging.getLogger("bluesky.RE.msg").info("Step 3: Prepare raster grid")
    grid = prepare_raster_grid(
        camera, motor_x, motor_z, plot=plot
    )
    # Step 4: Raster scan
    logging.getLogger("bluesky.RE.msg").info("Step 4: Raster scan")
    yield from grid_scan(
        [detector],
        motor_z,
        grid.initial_pos_z,
        grid.final_pos_z,
        number_of_steps_z,
        motor_x,
        grid.initial_pos_x,
        grid.final_pos_x,
        numer_of_steps_x,
        md=md,
    )

    # Steps 5 and 6: Find crystal and 2D centering
    logging.getLogger("bluesky.RE.msg").info(
        "Steps 5 and 6: Find crystal and 2D centering"
    )
    crystal_position, last_id = find_crystal_position(md["sample_id"], last_id=0)
    logging.getLogger("bluesky.RE.msg").info(
        "Max number of spots:", crystal_position.number_of_spots
    )
    logging.getLogger("bluesky.RE.msg").info(
        "Motor X position:", crystal_position.bluesky_event_doc.data.testrig_x
    )
    logging.getLogger("bluesky.RE.msg").info(
        "Motor Z position:", crystal_position.bluesky_event_doc.data.testrig_z
    )
    yield from mv(
        motor_x,
        crystal_position.bluesky_event_doc.data.testrig_x,
        motor_z,
        crystal_position.bluesky_event_doc.data.testrig_z,
    )

    # Step 7: Vertical scan
    logging.getLogger("bluesky.RE.msg").info("Step 7: Vertical scan")
    yield from mvr(motor_phi, 90)
    horizontal_grid = prepare_raster_grid(
        camera, motor_x, plot=plot
    )
    yield from grid_scan(
        [detector],
        motor_x,
        horizontal_grid.initial_pos_x,
        horizontal_grid.final_pos_x,
        numer_of_steps_x,
        md=md,
    )
    crystal_position, last_id = find_crystal_position(md["sample_id"], last_id=last_id)
    logging.getLogger("bluesky.RE.msg").info(
        f"Max number of spots: {crystal_position.number_of_spots}"
    )
    logging.getLogger("bluesky.RE.msg").info(
        f"Motor X position: {crystal_position.bluesky_event_doc.data.testrig_x}"
    )
    logging.getLogger("bluesky.RE.msg").info(
        f"Motor Z position: {crystal_position.bluesky_event_doc.data.testrig_z}"
    )
    yield from mv(
        motor_x,
        crystal_position.bluesky_event_doc.data.testrig_x,
    )

def test_scan(
    detector: DectrisDetector,
    motor_x: CosylabMotor,
    numer_of_steps_x: int,
    motor_y: CosylabMotor,
    motor_z: CosylabMotor,
    number_of_steps_z: int,
    motor_phi: CosylabMotor,
    camera: BlackFlyCam,
    md: dict,
    plot: bool = False,
    auto_focus: bool = True,
    min_focus: float = 0.0,
    max_focus: float = 1.0,
    tol: float = 0.3,
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
    numer_of_steps_x : int
        Number of steps (X axis)
    motor_y : CosylabMotor
        Motor Y
    motor_z : CosylabMotor
        Motor Z
    numer_of_steps_z : int
        Number of steps (Z axis)
    motor_phi : CosylabMotor
        Motor Phi
    camera : BlackFlyCam
        Camera
    md : dict
        Bluesky metadata, generally we include here the sample id,
        e.g. {"sample_id": "test_sample"}
    plot : bool, optional
        If true, we take snapshots of the plan at different stages for debugging purposes.
        By default false
    auto_focus : bool, optional
        If true, we autofocus the image before analysing an image with Lucid3,
        by default True
    min_focus : float, optional
        Minimum value to search for the maximum of var( Img * L(x,y) ),
        by default 0
    max_focus : float, optional
        Maximum value to search for the maximum of var( Img * L(x,y) ),
        by default 1
    tol : float, optional
        The tolerance used by the Golden-section search, by default 0.3

    Yields
    ------
    Generator[Msg, None, None]
        A bluesky plan tha centers the a sample using optical and X-ray centering
    """

    # Step 2: Loop centering
    """
    logging.getLogger("bluesky.RE.msg").info("Step 2: Loop centering")
    yield from optical_centering(
        motor_x,
        motor_y,
        motor_z,
        motor_phi,
        camera,
        plot,
        auto_focus,
        min_focus,
        max_focus,
        tol,
    )
    """

    # Step 3: Prepare raster grid
    logging.getLogger("bluesky.RE.msg").info("Step 3: Prepare raster grid")
    grid = prepare_raster_grid(
        camera, motor_x, motor_z, plot=plot
    )
    # Step 4: Raster scan
    logging.getLogger("bluesky.RE.msg").info("Step 4: Raster scan")
    yield from grid_scan(
        [detector],
        motor_z,
        grid.initial_pos_z,
        grid.final_pos_z,
        number_of_steps_z,
        motor_x,
        grid.initial_pos_x,
        grid.final_pos_x,
        numer_of_steps_x,
        md=md,
    )
