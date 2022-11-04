import ast
import logging
import os
import pickle
from os import environ
from time import sleep
from typing import Generator, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import redis
import requests
from bluesky.plan_stubs import mv
from bluesky.utils import Msg

from mx3_beamline_library.devices.classes.detectors import BlackFlyCam, DectrisDetector
from mx3_beamline_library.devices.classes.motors import CosylabMotor
from mx3_beamline_library.plans.basic_scans import grid_scan
from mx3_beamline_library.plans.optical_centering import optical_centering
from mx3_beamline_library.plans.psi_optical_centering import loopImageProcessing
from mx3_beamline_library.schemas.optical_and_xray_centering import (
    BlueskyEventDoc,
    RasterGridMotorCoordinates,
    SpotfinderResults,
    TestrigEventData,
)

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)

REDIS_HOST = environ.get("REDIS_HOST", "0.0.0.0")
REDIS_PORT = int(environ.get("REDIS_PORT", "6379"))

redis_connection = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)

BEAM_POSITION = ast.literal_eval(environ.get("BEAM_POSITION", "[640, 512]"))
PIXELS_PER_MM_X = float(environ.get("PIXELS_PER_MM_X", "292.87"))
PIXELS_PER_MM_Z = float(environ.get("PIXELS_PER_MM_Z", "292.87"))
MXCUBE_URL = environ.get("MXCUBE_URL", "http://localhost:8090")


def optical_and_xray_centering(
    detector: DectrisDetector,
    motor_x: CosylabMotor,
    number_of_steps_x: int,
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
    This is the plan that we call to run optical and x ray centering.
    It based on the procedure defined in Fig. 2 of
    Hirata et al. (2019). Acta Cryst. D75, 138-150.
    Currently the plan is implemented up to step 6 of
    Fig.2 in Hirata et a. (2019)

    Parameters
    ----------
    detector: DectrisDetector
        The dectris detector ophyd device
    motor_x : CosylabMotor
        Motor X
    number_of_steps_x : int
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
    logger.info("Step 2: Loop centering")
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
    logger.info("Step 3: Prepare raster grid")
    grid, rectangle_coordinates_in_pixels = prepare_raster_grid(
        camera, motor_x, motor_z, plot=plot
    )
    draw_grid_in_mxcube(
        rectangle_coordinates_in_pixels, number_of_steps_x, number_of_steps_z
    )
    # FIXME: Remove the following sleep, it is used for testing purposes only
    sleep(2)
    positions_before_grid_scan = {
        "motor_x": motor_x.position,
        "motor_z": motor_z.position,
    }
    # Step 4: Raster scan
    logger.info("Step 4: Raster scan")
    yield from grid_scan(
        [detector],
        motor_z,
        grid.initial_pos_z,
        grid.final_pos_z,
        number_of_steps_z,
        motor_x,
        grid.initial_pos_x,
        grid.final_pos_x,
        number_of_steps_x,
        md=md,
    )
    # Move the motors back to the original position to draw the grid
    yield from mv(
        motor_x,
        positions_before_grid_scan["motor_x"],
        motor_z,
        positions_before_grid_scan["motor_z"],
    )

    # Steps 5 and 6: Find crystal and 2D centering
    logger.info("Steps 5 and 6: Find crystal and 2D centering")
    crystal_position, last_id, nummber_of_spots_list = find_crystal_position(
        md["sample_id"], last_id=0
    )
    logger.info(f"Max number of spots: {crystal_position.number_of_spots}")
    logger.info(
        f"Motor X position: {crystal_position.bluesky_event_doc.data.testrig_x}"
    )
    logger.info(
        f"Motor Z position: {crystal_position.bluesky_event_doc.data.testrig_z}"
    )
    draw_grid_in_mxcube(
        rectangle_coordinates_in_pixels,
        number_of_steps_x,
        number_of_steps_z,
        number_of_spots_list=nummber_of_spots_list,
    )

    """
    yield from mv(
        motor_x,
        crystal_position.bluesky_event_doc.data.testrig_x,
        motor_z,
        crystal_position.bluesky_event_doc.data.testrig_z,
    )

    # Step 7: Vertical scan
    logger.info("Step 7: Vertical scan")
    yield from mvr(motor_phi, 90)
    horizontal_grid = prepare_raster_grid(camera, motor_x, plot=plot)
    yield from grid_scan(
        [detector],
        motor_x,
        horizontal_grid.initial_pos_x,
        horizontal_grid.final_pos_x,
        number_of_steps_x,
        md=md,
    )
    crystal_position, last_id = find_crystal_position(md["sample_id"], last_id=last_id)
    logger.info(
        f"Max number of spots: {crystal_position.number_of_spots}"
    )
    logger.info(
        f"Motor X position: {crystal_position.bluesky_event_doc.data.testrig_x}"
    )
    logger.info(
        f"Motor Z position: {crystal_position.bluesky_event_doc.data.testrig_z}"
    )
    yield from mv(
        motor_x,
        crystal_position.bluesky_event_doc.data.testrig_x,
    )
    """


def prepare_raster_grid(
    camera: BlackFlyCam,
    motor_x: CosylabMotor,
    motor_z: CosylabMotor,
    plot: bool = False,
) -> tuple[RasterGridMotorCoordinates, dict]:
    """
    Prepares a raster grid. The limits of the grid are obtained using
    the PSI loop centering code

    Parameters
    ----------
    camera : BlackFlyCam
        Camera
    motor_x : CosylabMotor
        Motor X
    motor_z : CosylabMotor
        Motor Z
    plot : bool, optional
        If True, we plot the raster grid and save it to a file, by default False

    Returns
    -------
    motor_coordinates : RasterGridMotorCoordinates
        A pydantinc model containing the initial and final motor positions of the grid.
    rectangle_coordinates : dict
        Rectangle coordinates in pixels
    """

    array_data: npt.NDArray = camera.array_data.get()
    data = array_data.reshape(
        camera.height.get(), camera.width.get(), camera.depth.get()
    ).astype(
        np.uint8
    )  # the code only works with np.uint8 data types

    procImg = loopImageProcessing(data)
    procImg.findContour(zoom="-208.0", beamline="X06DA")
    procImg.findExtremes()
    rectangle_coordinates = procImg.fitRectangle()

    if plot:
        plot_raster_grid(
            camera,
            rectangle_coordinates,
            "step_3_prep_raster",
        )

    # Z motor positions
    initial_pos_z_pixels = abs(rectangle_coordinates["top_left"][1] - BEAM_POSITION[1])
    final_pos_z_pixels = abs(
        rectangle_coordinates["bottom_right"][1] - BEAM_POSITION[1]
    )

    initial_pos_z = motor_z.position - initial_pos_z_pixels / PIXELS_PER_MM_Z
    final_pos_z = motor_z.position + final_pos_z_pixels / PIXELS_PER_MM_Z

    # X motor positions
    initial_pos_x_pixels = abs(rectangle_coordinates["top_left"][0] - BEAM_POSITION[0])
    final_pos_x_pixels = abs(
        rectangle_coordinates["bottom_right"][0] - BEAM_POSITION[0]
    )

    initial_pos_x = motor_x.position - initial_pos_x_pixels / PIXELS_PER_MM_X
    final_pos_x = motor_x.position + final_pos_x_pixels / PIXELS_PER_MM_X

    motor_coordinates = RasterGridMotorCoordinates(
        initial_pos_x=initial_pos_x,
        final_pos_x=final_pos_x,
        initial_pos_z=initial_pos_z,
        final_pos_z=final_pos_z,
    )
    logger.info(f"Raster grid coordinates [mm]: {motor_coordinates}")

    return motor_coordinates, rectangle_coordinates


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
) -> tuple[SpotfinderResults, bytes, list]:
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
    spotfinder_results, last_id, number_of_spots_list : tuple[SpotfinderResults, bytes, list]
        SpotfinderResults corresponding to the maximum number of spots,
        the redis_streams last_id, and a list containing the number of spots in the grid
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
    logger.info(f"Max number of spots: {number_of_spots_list[argmax]}")
    logger.info(f"Metadata associated with the max number of spots: {result[argmax]}")
    return result[argmax], last_id, number_of_spots_list


def draw_grid_in_mxcube(
    rectangle_coordinates: dict,
    num_cols: int,
    num_rows: int,
    grid_id: int = 1,
    number_of_spots_list: Optional[list[int]] = None,
):
    """Draws a grid in mxcube

    Parameters
    ----------
    rectangle_coordinates : dict
        Rectangle coodinates of the grid obtained from the PSI loop centering code
    num_cols : int
        Number of columns of the grid
    num_rows : int
        Number of rows of the grid
    grid_id : int, optional
        Grid id used by MXCuBE, by default 1
    number_of_spots_list : list[int], optional
        List containing the number of spots of the grid, by default None

    Returns
    -------
    None
    """
    if number_of_spots_list is not None:
        heatmap_and_crystalmap = create_heatmap_and_crystal_map(
            num_cols, num_rows, number_of_spots_list
        )
    else:
        heatmap_and_crystalmap = []

    width = int(
        rectangle_coordinates["bottom_right"][0] - rectangle_coordinates["top_left"][0]
    )  # pixels
    height = int(
        rectangle_coordinates["bottom_right"][1] - rectangle_coordinates["top_left"][1]
    )  # pixels

    mm_per_pixel = 1 / PIXELS_PER_MM_X
    cell_width = (width / num_cols) * mm_per_pixel * 1000  # micrometers
    cell_height = (height / num_rows) * mm_per_pixel * 1000  # micrometers

    # TODO: Maybe it'd be useful to pass accurate info about the motor positions
    # even though it is techincally not used to draw the grid
    mxcube_payload = {
        "shapes": [
            {
                "cellCountFun": "zig-zag",
                "cellHSpace": 0,
                "cellHeight": cell_height,
                "cellVSpace": 0,
                "cellWidth": cell_width,
                "height": height,
                "width": width,
                "hideThreshold": 5,
                "id": f"G{grid_id}",
                "label": "Grid",
                "motorPositions": {
                    "beamX": 0.141828,
                    "beamY": 0.105672,
                    "kappa": 11,
                    "kappaPhi": 22,
                    "phi": 311.1,
                    "phiy": 34.30887849323582,
                    "phiz": 1.1,
                    "sampx": -0.0032739045158179936,
                    "sampy": -1.0605072324693783,
                },
                "name": f"Grid-{grid_id}",
                "numCols": num_cols,
                "numRows": num_rows,
                "result": heatmap_and_crystalmap,
                "screenCoord": rectangle_coordinates["top_left"].tolist(),
                "selected": True,
                "state": "SAVED",
                "t": "G",
                "pixelsPerMm": [PIXELS_PER_MM_X, PIXELS_PER_MM_Z],
                # 'dxMm': 1/292.8705182537115,
                # 'dyMm': 1/292.8705182537115
            }
        ]
    }

    try:
        response = requests.post(
            os.path.join(MXCUBE_URL, "mxcube/api/v0.1/sampleview/shapes/create_grid"),
            json=mxcube_payload,
        )
        logger.info(f"MXCuBE response: {response.json()}")

    except requests.exceptions.ConnectionError:
        logger.info("MXCuBE is not available, cannot draw grid in MXCuBE")


def create_heatmap_and_crystal_map(
    num_cols: int, num_rows: int, number_of_spots_list: list[int]
) -> npt.NDArray:
    """
    Creates a heatmap from the number of spots, number of columns
    and number of rows of a grid.

    Parameters
    ----------
    num_cols : int
        Number of columns
    num_rows : int
        Number of rows
    number_of_spots_list : list[int]
        List containing number of spots

    Returns
    -------
    dict
        A dictionary containing a heatmap and crystal map in rbga format.
        Currently the crystal map is a random array and only the heatmap
        contains meaningful results
    """

    x = np.arange(num_cols)
    y = np.arange(num_rows)

    y, x = np.meshgrid(x, y)
    z = np.array([number_of_spots_list]).reshape(num_rows, num_cols)

    z_min = np.min(z)
    z_max = np.max(z)

    _, ax = plt.subplots()

    heatmap = ax.pcolormesh(x, y, z, cmap="seismic", vmin=z_min, vmax=z_max)
    heatmap = heatmap.to_rgba(z, norm=True).reshape(num_cols * num_rows, 4)

    # The following could probably be done more efficiently without using for loops
    heatmap_array = np.ones(heatmap.shape)
    for i in range(num_rows * num_cols):
        for j in range(4):
            if heatmap[i][j] != 1.0:
                heatmap_array[i][j] = int(heatmap[i][j] * 255)

    heatmap_array = heatmap_array.tolist()

    heatmap = {}
    crystalmap = {}

    for i in range(1, num_rows * num_cols + 1):
        heatmap[i] = [i, list(heatmap_array[i - 1])]

        crystalmap[i] = [
            i,
            [
                int(np.random.random() * 255),
                int(np.random.random() * 255),
                int(np.random.random() * 255),
                1,
            ],
        ]

    return {"heatmap": heatmap, "crystalmap": crystalmap}


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
