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
from bluesky.plan_stubs import mv, mvr
from bluesky.utils import Msg
from pydantic import ValidationError

from mx3_beamline_library.devices.classes.detectors import BlackFlyCam, DectrisDetector
from mx3_beamline_library.devices.classes.motors import CosylabMotor
from mx3_beamline_library.plans.basic_scans import grid_scan
from mx3_beamline_library.plans.optical_centering import OpticalCentering
from mx3_beamline_library.schemas.optical_and_xray_centering import (
    BlueskyEventDoc,
    RasterGridMotorCoordinates,
    SpotfinderResults,
    TestrigEventData,
)
from mx3_beamline_library.science.optical_and_loop_centering.crystal_finder import (
    CrystalFinder,
    CrystalFinder3D,
)
from mx3_beamline_library.science.optical_and_loop_centering.psi_optical_centering import (
    loopImageProcessing,
)

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)


class OpticalAndXRayCentering(OpticalCentering):
    """
    This plan consists of different steps explained as follows:
    First, we center the loop using the methods described in the OpticalCentering class.
    Once the loop has been centered, we prepare a raster grid, execute a raster scan, and find
    the crystals using the CrystalFinder.
    Then, we rotate the loop 90 degrees and repeat this process to determine the
    location of the crystals in the orthogonal plane, which allows us to finally
    infer the positions of the crystals in 3D.
    """

    def __init__(
        self,
        detector: DectrisDetector,
        camera: BlackFlyCam,
        motor_x: CosylabMotor,
        number_of_steps_x: int,
        motor_y: CosylabMotor,
        motor_z: CosylabMotor,
        number_of_steps_z: int,
        motor_phi: CosylabMotor,
        md: dict,
        beam_position: tuple[int, int],
        pixels_per_mm_x: float,
        pixels_per_mm_z: float,
        threshold: float,
        auto_focus: bool = True,
        min_focus: float = 0,
        max_focus: float = 1,
        tol: float = 0.3,
        method: str = "psi",
        plot: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        detector: DectrisDetector
            The dectris detector ophyd device
        camera : BlackFlyCam
            Camera
        motor_x : CosylabMotor
            Motor X
        number_of_steps_x : int
            Number of steps (X axis)
        motor_y : CosylabMotor
            Motor Y
        motor_z : CosylabMotor
            Motor Z
        number_of_steps_z : int
            Number of steps (Z axis)
        motor_phi : CosylabMotor
            Motor Phi
        md : dict
            Bluesky metadata, we include here the sample id,
            e.g. {"sample_id": "test_sample"}
        beam_position : tuple[int, int]
            Position of the beam
        pixels_per_mm_x : float
            Pixels per mm x
        pixels_per_mm_z : float
            Pixels per mm z
        threshold : float
            This parameter is used by the CrystalFinder class. Below this threshold,
            we replace all numbers of the number_of_spots array obtained from
            the grid scan plan with zeros.
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
        method : str, optional
            Method used to find the edge of the loop. Can be either
            psi or lucid, by default "psi"
        plot : bool, optional
            If true, we take snapshots of the plan at different stages for debugging purposes.
            By default false

        Returns
        -------
        None
        """
        super().__init__(
            camera,
            motor_x,
            motor_y,
            motor_z,
            motor_phi,
            beam_position,
            pixels_per_mm_x,
            pixels_per_mm_z,
            auto_focus,
            min_focus,
            max_focus,
            tol,
            method,
            plot,
        )

        self.detector = detector
        self.number_of_steps_x = number_of_steps_x
        self.number_of_steps_z = number_of_steps_z
        self.md = md
        self.threshold = threshold

        REDIS_HOST = environ.get("REDIS_HOST", "0.0.0.0")
        REDIS_PORT = int(environ.get("REDIS_PORT", "6379"))
        self.redis_connection = redis.StrictRedis(
            host=REDIS_HOST, port=REDIS_PORT, db=0
        )

        self.mxcube_url = environ.get("MXCUBE_URL", "http://localhost:8090")

    def start(self) -> Generator[Msg, None, None]:
        """
        This is the plan that we call to run optical and x ray centering.
        It based on the centering code defined in Fig. 2 of
        Hirata et al. (2019). Acta Cryst. D75, 138-150.

        Yields
        ------
        Generator[Msg, None, None]
            A bluesky plan tha centers the a sample using optical and X-ray centering
        """

        # Step 2: Loop centering
        logger.info("Step 2: Loop centering")
        yield from self.center_loop()

        positions_before_grid_scan = {
            "motor_x": self.motor_x.position,
            "motor_z": self.motor_z.position,
        }

        # Steps 3 to 6:
        logger.info(
            "Steps 3-6: Prepare raster grid, execute raster scan, and find crystals"
        )
        (
            centers_of_mass_flat,
            crystal_locations_flat,
            distances_between_crystals_flat,
            last_id,
        ) = yield from self.start_raster_scan_and_find_crystals(
            positions_before_grid_scan, last_id=0, filename="flat"
        )

        # Step 7: Rotate loop 90 degrees, repeat steps 3 to 6
        logger.info("Step 7: Rotate loop 90 degrees, repeat steps 3 to 6")
        yield from mvr(self.motor_phi, 90)

        (
            centers_of_mass_edge,
            crystal_locations_edge,
            distances_between_crystals_edge,
            last_id,
        ) = yield from self.start_raster_scan_and_find_crystals(
            positions_before_grid_scan, last_id=last_id, filename="edge"
        )

        # Step 8: Infer location of crystals in 3D
        logger.info("Step 8: Infer location of crystals in 3D")
        crystal_finder_3d = CrystalFinder3D(
            crystal_locations_flat,
            crystal_locations_edge,
            centers_of_mass_flat,
            centers_of_mass_edge,
            distances_between_crystals_flat,
            distances_between_crystals_edge,
        )
        crystal_finder_3d.plot_crystals(plot_centers_of_mass=True, save=self.plot)

    def start_raster_scan_and_find_crystals(
        self,
        positions_before_grid_scan: dict,
        last_id: Union[int, bytes],
        filename: str = "crystal_finder_results",
    ) -> tuple[list[tuple[int, int]], list[dict], list[dict[str, int]], bytes]:
        """
        Prepares the raster grid, executes the raster plan, and finds the crystals
        in a loop using the CrystalFinder. This method is can be reused to analyse the
        flat and edge surfaces of the loop.

        Parameters
        ----------
        positions_before_grid_scan : dict
            A dictionary containing the motor positions before the start of the plan
        last_id : Union[int, bytes]
            Redis streams last_id
        filename : str, optional
            Name of the file used to save the results if self.plot=True,
            by default "crystal_finder_results"

        Returns
        -------
        tuple[list[tuple[int, int]], list[dict], list[dict[str, int]], bytes]
            A list containing the centers of mass of all crystals in the loop,
            a list of dictionaries containing information about the locations and sizes
            of all crystals,
            a list of dictionaries describing the distance between all overlapping crystals,
            and the updated redis streams last_id

        """
        grid, rectangle_coordinates_in_pixels = self.prepare_raster_grid(
            f"step_3_prep_raster_{filename}"
        )
        self.draw_grid_in_mxcube(
            rectangle_coordinates_in_pixels,
            self.number_of_steps_x,
            self.number_of_steps_z,
        )
        # FIXME: The following sleep statement is for testing purposes only
        sleep(2)
        # Step 4: Raster scan
        logger.info("Starting raster scan...")
        yield from grid_scan(
            [self.detector],
            self.motor_z,
            grid.initial_pos_z,
            grid.final_pos_z,
            self.number_of_steps_z,
            self.motor_x,
            grid.initial_pos_x,
            grid.final_pos_x,
            self.number_of_steps_x,
            md=self.md,
        )
        # Move the motors back to the original position to draw the grid
        yield from mv(
            self.motor_x,
            positions_before_grid_scan["motor_x"],
            self.motor_z,
            positions_before_grid_scan["motor_z"],
        )

        # Find crystals
        logger.info("Finding crystals")
        (
            centers_of_mass,
            crystal_locations,
            distances_between_crystals,
            number_of_spots_array,
            last_id,
        ) = self.find_crystal_positions(
            self.md["sample_id"],
            last_id=last_id,
            n_rows=self.number_of_steps_z,
            n_cols=self.number_of_steps_x,
            filename=filename,
        )

        self.draw_grid_in_mxcube(
            rectangle_coordinates_in_pixels,
            self.number_of_steps_x,
            self.number_of_steps_z,
            number_of_spots_array=number_of_spots_array,
        )

        return (  # noqa
            centers_of_mass,
            crystal_locations,
            distances_between_crystals,
            last_id,
        )

    def prepare_raster_grid(
        self, filename: str = "step_3_prep_raster"
    ) -> tuple[RasterGridMotorCoordinates, dict]:
        """
        Prepares a raster grid. The limits of the grid are obtained using
        the PSI loop centering code

        Parameters
        ----------
        filename : str
            Name of the file used to plot save the results if self.plot=True,
            by default step_3_prep_raster

        Returns
        -------
        motor_coordinates : RasterGridMotorCoordinates
            A pydantic model containing the initial and final motor positions of the grid.
        rectangle_coordinates : dict
            Rectangle coordinates in pixels
        """
        # the loopImageProcessing code only works with np.uint8 data types
        data = self.get_image_from_camera(np.uint8)

        procImg = loopImageProcessing(data)
        procImg.findContour(zoom="-208.0", beamline="X06DA")
        procImg.findExtremes()
        rectangle_coordinates = procImg.fitRectangle()

        if self.plot:
            self.plot_raster_grid(
                rectangle_coordinates,
                filename,
            )

        # Z motor positions
        initial_pos_z_pixels = abs(
            rectangle_coordinates["top_left"][1] - self.beam_position[1]
        )
        final_pos_z_pixels = abs(
            rectangle_coordinates["bottom_right"][1] - self.beam_position[1]
        )

        initial_pos_z = (
            self.motor_z.position - initial_pos_z_pixels / self.pixels_per_mm_z
        )
        final_pos_z = self.motor_z.position + final_pos_z_pixels / self.pixels_per_mm_z

        # X motor positions
        initial_pos_x_pixels = abs(
            rectangle_coordinates["top_left"][0] - self.beam_position[0]
        )
        final_pos_x_pixels = abs(
            rectangle_coordinates["bottom_right"][0] - self.beam_position[0]
        )

        initial_pos_x = (
            self.motor_x.position - initial_pos_x_pixels / self.pixels_per_mm_x
        )
        final_pos_x = self.motor_x.position + final_pos_x_pixels / self.pixels_per_mm_x

        motor_coordinates = RasterGridMotorCoordinates(
            initial_pos_x=initial_pos_x,
            final_pos_x=final_pos_x,
            initial_pos_z=initial_pos_z,
            final_pos_z=final_pos_z,
        )
        logger.info(f"Raster grid coordinates [mm]: {motor_coordinates}")

        return motor_coordinates, rectangle_coordinates

    def find_crystal_positions(
        self,
        sample_id: str,
        last_id: Union[int, bytes],
        n_rows: int,
        n_cols: int,
        filename: str = "crystal_finder_results",
    ) -> tuple[
        list[tuple[int, int]], list[dict], list[dict[str, int]], npt.NDArray, bytes
    ]:
        """
        Finds the crystal position based on the number of spots obtained from a
        grid_scan using the CrystalFinder class. The number of spots are obtained
        from redis streams, which are generated by the mx-spotfinder in the
        mx-zmq-consumer.

        Parameters
        ----------
        sample_id : str
            Sample id
        last_id : Union[int, bytes]
            Redis streams last_id
        n_rows : int
            Number of rows of the grid
        n_cols : int
            Number of columns of the grid
        filename : str, optional
            The name of the file used to save the CrystalFinder results if
            self.plot=True, by default crystal_finder_results

        Returns
        -------
        tuple[list[tuple[int, int]], list[dict], list[dict[str, int]], npt.NDArray, bytes]
            A list containing the centers of mass of all crystals in the loop,
            a list of dictionaries containing information about the locations and sizes
            of all crystals,
            a list of dictionaries describing the distance between all overlapping crystals,
            a numpy array containing the numbers of spots, which shape is (n_rows, n_cols),
            and the last id of the redis streams sequence
        """
        result = []
        number_of_spots_list = []
        for _ in range(self.redis_connection.xlen(sample_id)):
            try:
                spotfinder_results, last_id = self.read_message_from_redis_streams(
                    sample_id, last_id
                )
                result.append(spotfinder_results)
                number_of_spots_list.append(spotfinder_results.number_of_spots)
            except IndexError:
                pass

        spots_array = np.array(number_of_spots_list).reshape(n_rows, n_cols)
        spotfinder_results_array = np.array(result).reshape(n_rows, n_cols)

        crystal_finder = CrystalFinder(spots_array, threshold=self.threshold)
        (
            centers_of_mass,
            crystal_locations,
            distances_between_crystals,
        ) = crystal_finder.plot_crystal_finder_results(
            save=self.plot, filename=filename
        )

        for cm in centers_of_mass:
            logger.info(
                "Spotfinder results evaluated at the centers of mass: "
                f"{spotfinder_results_array[cm[1]][cm[0]]}"
            )

        # TODO: the spotfinder_results_array is useful for relating pixels to
        # actual motor positions since it contains all the parameters described in the
        # SpotfinderResults pydantic model. At this stage, we're just interested in
        # finding the position of the crystal in units of pixels, so we
        # leave translation from pixels to motor positions for the next sprint

        return (
            centers_of_mass,
            crystal_locations,
            distances_between_crystals,
            spots_array,
            last_id,
        )

    def read_message_from_redis_streams(
        self, topic: str, id: Union[bytes, int]
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
        response = self.redis_connection.xread({topic: id}, count=1)

        # Extract key and messages from the response
        _, messages = response[0]

        # Update last_id and store messages data
        last_id, data = messages[0]

        try:
            bluesky_event_doc = BlueskyEventDoc.parse_obj(
                pickle.loads(data[b"bluesky_event_doc"])
            )
            bluesky_event_doc.data = TestrigEventData.parse_obj(bluesky_event_doc.data)
        except ValidationError:
            # This is used only when kafka is not available, intended
            # for testing purposes only
            bluesky_event_doc = pickle.loads(data[b"bluesky_event_doc"])

        spotfinder_results = SpotfinderResults(
            type=data[b"type"],
            number_of_spots=data[b"number_of_spots"],
            image_id=data[b"image_id"],
            sequence_id=data[b"sequence_id"],
            bluesky_event_doc=bluesky_event_doc,
        )

        sequence_id_zmq = spotfinder_results.sequence_id

        try:
            sequence_id_bluesky_doc = (
                spotfinder_results.bluesky_event_doc.data.dectris_detector_sequence_id
            )
        except AttributeError:
            # This is used only when kafka is not available, intended
            # for testing purposes only
            sequence_id_bluesky_doc = None

        if sequence_id_bluesky_doc is not None:
            assert sequence_id_zmq == sequence_id_bluesky_doc, (
                "Sequence_id obtained from bluesky doc is different from the ZMQ sequence_id "
                f"sequence_id_zmq: {sequence_id_zmq}, "
                f"sequence_id_bluesky_doc: {sequence_id_bluesky_doc}"
            )

        return spotfinder_results, last_id

    def draw_grid_in_mxcube(
        self,
        rectangle_coordinates: dict,
        num_cols: int,
        num_rows: int,
        grid_id: int = 1,
        number_of_spots_array: Optional[npt.NDArray] = None,
    ):
        """Draws a grid in mxcube

        Parameters
        ----------
        rectangle_coordinates : dict
            Rectangle coordinates of the grid obtained from the PSI loop centering code
        num_cols : int
            Number of columns of the grid
        num_rows : int
            Number of rows of the grid
        grid_id : int, optional
            Grid id used by MXCuBE, by default 1
        number_of_spots_array : npt.NDArray, optional
            A numpy array of shape (n_rows, n_cols) containing the
            number of spots of the grid, by default None

        Returns
        -------
        None
        """
        if number_of_spots_array is not None:
            heatmap_and_crystalmap = self.create_heatmap_and_crystal_map(
                num_cols, num_rows, number_of_spots_array
            )
        else:
            heatmap_and_crystalmap = []

        width = int(
            rectangle_coordinates["bottom_right"][0]
            - rectangle_coordinates["top_left"][0]
        )  # pixels
        height = int(
            rectangle_coordinates["bottom_right"][1]
            - rectangle_coordinates["top_left"][1]
        )  # pixels

        mm_per_pixel = 1 / self.pixels_per_mm_x
        cell_width = (width / num_cols) * mm_per_pixel * 1000  # micrometers
        cell_height = (height / num_rows) * mm_per_pixel * 1000  # micrometers

        # TODO: Maybe it'd be useful to pass accurate info about the motor positions
        # even though it is technically not used to draw the grid
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
                    "pixelsPerMm": [self.pixels_per_mm_x, self.pixels_per_mm_z],
                    # 'dxMm': 1/292.8705182537115,
                    # 'dyMm': 1/292.8705182537115
                }
            ]
        }

        try:
            requests.post(
                os.path.join(
                    self.mxcube_url, "mxcube/api/v0.1/sampleview/shapes/create_grid"
                ),
                json=mxcube_payload,
            )
            logger.info("MXCuBE request posted successfully")

        except requests.exceptions.ConnectionError:
            logger.info("MXCuBE is not available, cannot draw grid in MXCuBE")

    def create_heatmap_and_crystal_map(
        self, num_cols: int, num_rows: int, number_of_spots_array: npt.NDArray
    ) -> dict:
        """
        Creates a heatmap from the number of spots, number of columns
        and number of rows of a grid. The crystal map currently returns
        random numbers since at this stage we only calculate the heatmap.

        Parameters
        ----------
        num_cols : int
            Number of columns
        num_rows : int
            Number of rows
        number_of_spots_array: npt.NDArray
            A numpy array of shape (n_rows, n_cols) containing the
            number of spots of the grid

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

        z_min = np.min(number_of_spots_array)
        z_max = np.max(number_of_spots_array)

        _, ax = plt.subplots()

        heatmap = ax.pcolormesh(
            x, y, number_of_spots_array, cmap="seismic", vmin=z_min, vmax=z_max
        )
        heatmap = heatmap.to_rgba(number_of_spots_array, norm=True).reshape(
            num_cols * num_rows, 4
        )

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
        self,
        rectangle_coordinates: dict,
        filename: str,
    ) -> None:
        """
        Plots the limits of the raster grid on top of the image taken from the
        camera.

        Parameters
        ----------
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
        data = self.get_image_from_camera()
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


def optical_and_xray_centering(
    detector: DectrisDetector,
    camera: BlackFlyCam,
    motor_x: CosylabMotor,
    number_of_steps_x: int,
    motor_y: CosylabMotor,
    motor_z: CosylabMotor,
    number_of_steps_z: int,
    motor_phi: CosylabMotor,
    md: dict,
    beam_position: tuple[int, int],
    pixels_per_mm_x: float,
    pixels_per_mm_z: float,
    threshold: float,
    auto_focus: bool = True,
    min_focus: float = 0,
    max_focus: float = 1,
    tol: float = 0.3,
    method: str = "psi",
    plot: bool = False,
) -> Generator[Msg, None, None]:
    """
    This is just a wrapper to execute the optical and xray centering plan
    using the OpticalAndXRayCentering class. This function is needed because the
    bluesky-queueserver does not interact nicely with classes.

    Parameters
    ----------
    detector: DectrisDetector
        The dectris detector ophyd device
    camera : BlackFlyCam
        Camera
    motor_x : CosylabMotor
        Motor X
    number_of_steps_x : int
        Number of steps (X axis)
    motor_y : CosylabMotor
        Motor Y
    motor_z : CosylabMotor
        Motor Z
    number_of_steps_z : int
        Number of steps (Z axis)
    motor_phi : CosylabMotor
        Motor Phi
    md : dict
        Bluesky metadata, we include here the sample id,
        e.g. {"sample_id": "test_sample"}
    beam_position : tuple[int, int]
        Position of the beam
    pixels_per_mm_x : float
        Pixels per mm x
    pixels_per_mm_z : float
        Pixels per mm z
    threshold : float
        This parameter is used by the CrystalFinder class. Below this threshold,
        we replace all numbers of the number_of_spots array obtained from
        the grid scan plan with zeros.
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
    method : str, optional
        Method used to find the edge of the loop. Can be either
        psi or lucid, by default "psi"
    plot : bool, optional
        If true, we take snapshots of the plan at different stages for debugging purposes.
        By default false

    Returns
    -------
    Generator[Msg, None, None]
    """

    _optical_and_xray_centering = OpticalAndXRayCentering(
        detector,
        camera,
        motor_x,
        number_of_steps_x,
        motor_y,
        motor_z,
        number_of_steps_z,
        motor_phi,
        md=md,
        beam_position=beam_position,
        pixels_per_mm_x=pixels_per_mm_x,
        pixels_per_mm_z=pixels_per_mm_z,
        threshold=threshold,
        auto_focus=auto_focus,
        min_focus=min_focus,
        max_focus=max_focus,
        tol=tol,
        method=method,
        plot=plot,
    )

    yield from _optical_and_xray_centering.start()
