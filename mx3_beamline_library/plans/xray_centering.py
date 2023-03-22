import asyncio
import logging
import os
import pickle
import time
from os import environ
from typing import Generator, Optional, Union

import httpx
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import redis
import yaml
from bluesky.plan_stubs import mv
from bluesky.preprocessors import monitor_during_wrapper, run_wrapper
from bluesky.utils import Msg
from ophyd import Signal

from ..devices.classes.detectors import (
    BlackFlyCam,
    DectrisDetector,
    MDRedisCam,
)
from ..devices.classes.motors import (
    CosylabMotor,
    MD3BackLight,
    MD3Motor,
    MD3Phase,
    MD3Zoom,
)
from ..plans.basic_scans import (
    arm_trigger_and_disarm_detector,
    md3_4d_scan,
    md3_grid_scan,
)
from ..schemas.crystal_finder import CrystalPositions
from ..schemas.xray_centering import (
    RasterGridMotorCoordinates,
    SpotfinderResults,
)
from ..schemas.optical_centering import CenteredLoopMotorCoordinates
from ..science.optical_and_loop_centering.crystal_finder import (
    CrystalFinder,
    CrystalFinder3D,
)
from ..science.optical_and_loop_centering.psi_optical_centering import (
    loopImageProcessing,
)

from ..schemas.xray_centering import MD3ScanResponse
from ..schemas.detector import UserData


logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)


class XRayCentering:
    """
    This plan consists of different steps explained as follows:
    1) Center the loop using the methods described in the OpticalCentering class.
    2) Prepare a raster grid, execute a raster scan for the surface of the loop where
    its area is largest
    3) Find the crystals using the CrystalFinder.
    4) Rotate the loop 90 degrees and repeat steps 2 and 3, to determine the
    location of the crystals in the orthogonal plane, which allows us to finally
    infer the positions of the crystals in 3D.
    """

    def __init__(
        self,
        sample_id: str,
        detector: DectrisDetector,
        omega: Union[CosylabMotor, MD3Motor],
        zoom: MD3Zoom,
        grid_scan_type: str,
        exposure_time: float = 2.0,
        omega_range: float = 0.0,
        count_time: float = None
    ) -> None:
        """
        Parameters
        ----------
        sample_id: str
            Sample id
        detector: DectrisDetector
            The dectris detector ophyd device
        omega : Union[CosylabMotor, MD3Motor]
            Omega
        zoom : MD3Zoom
            Zoom
        grid_scan_type: str
            Grid scan type, could be either `flat`, or `edge`.
        threshold : float
            This parameter is used by the CrystalFinder class. Below this threshold,
            we replace all numbers of the number_of_spots array obtained from
            the grid scan plan with zeros.
        exposure_time : float
            Detector exposure time
        omega_range : float, optional
            Omega range (degrees) for the scan, by default 0
        count_time : float
            Detector count time, by default None. If this parameter is not set, 
            it is set to frame_time - 0.0000001 by default. This calculation 
            is done via the DetectorConfiguration pydantic model.

        Returns
        -------
        None
        """
        self.sample_id = sample_id
        self.detector = detector
        self.omega = omega
        self.zoom = zoom
        self.grid_scan_type = grid_scan_type
        self.exposure_time = exposure_time
        self.omega_range = omega_range
        self.count_time = count_time

        REDIS_HOST = environ.get("REDIS_HOST", "0.0.0.0")
        REDIS_PORT = int(environ.get("REDIS_PORT", "6379"))
        self.redis_connection = redis.StrictRedis(
            host=REDIS_HOST, port=REDIS_PORT, db=0
        )

        self.mxcube_url = environ.get("MXCUBE_URL", "http://localhost:8090")

        self.grid_id = 0

        self.md3_scan_response = Signal(name="md3_scan_response", kind="normal")
        
        self.get_optical_centering_results()



    def get_optical_centering_results(self):
        results = pickle.loads(
            self.redis_connection.get(f"optical_centering_results:{self.sample_id}")
        )
        self.centered_loop_coordinates = CenteredLoopMotorCoordinates.parse_obj(
            results["centered_loop_coordinates"]
        )
        self.edge_angle = results["edge_angle"]
        self.flat_angle = results["flat_angle"]
        self.flat_grid_motor_coordinates = RasterGridMotorCoordinates.parse_obj(
            results["flat_grid_motor_coordinates"]
        )
        self.edge_grid_motor_coordinates = RasterGridMotorCoordinates.parse_obj(
            results["edge_grid_motor_coordinates"]
        )


    def start_grid_scan(self) -> Generator[Msg, None, None]:
        """
        Runs an edge or flat grid scan, depending on the value of self.grid_scan_type

        Yields
        ------
        Generator[Msg, None, None]
            A bluesky plan tha centers the a sample using optical and X-ray centering
        """


        if self.grid_scan_type.lower() == "flat":
            logger.info(f"Running grid scan: {self.grid_scan_type}")
            yield from mv(self.omega, self.flat_angle)

            yield from self._grid_scan(
                self.flat_grid_motor_coordinates
            )

        elif self.grid_scan_type.lower() == "edge":
            logger.info(f"Running grid scan: {self.grid_scan_type}")
            yield from mv(self.omega, self.edge_angle)

            yield from self._grid_scan(
                self.edge_grid_motor_coordinates,
            )

        """
        # Step 8: Infer location of crystals in 3D
        logger.info("Step 8: Infer location of crystals in 3D")
        crystal_finder_3d = CrystalFinder3D(
            crystal_locations_flat,
            crystal_locations_edge,
            distances_between_crystals_flat,
            distances_between_crystals_edge,
        )
        crystal_finder_3d.plot_crystals(plot_centers_of_mass=False, save=self.plot)
        """

    def _grid_scan(
        self,
        grid: RasterGridMotorCoordinates,
        draw_grid_in_mxcube: bool = False,
        rectangle_coordinates_in_pixels: dict = None,
    ) -> None:
        """
        Runs an md3_grid_scan or md3_4d_scan depending on the number of rows an columns

        Parameters
        ----------
        grid : RasterGridMotorCoordinates
            A RasterGridMotorCoordinates object which contains information about the
            raster grid, including its width, height and initial and final positions
            of sample_x, sample_y, and alignment_y
        draw_grid_in_mxcube : bool
            If true, we draw a grid in mxcube, by default False
        rectangle_coordinates_in_pixels : dict
            Rectangle coordinates in pixels

        Returns
        -------
        None

        """
        logger.info("Starting raster scan...")
        logger.info(f"Number of columns: {grid.number_of_columns}")
        logger.info(f"Number of rows: {grid.number_of_rows}")
        logger.info(f"Grid width [mm]: {grid.width}")
        logger.info(f"Grid height [mm]: {grid.height}")


        # NOTE: The md3_grid_scan does not like number_of_columns < 2. If
        # number_of_columns < 2 we use the md3_3d_scan instead, setting scan_range=0,
        # and keeping the values of sample_x, sample_y, and alignment_z constant
        user_data = UserData(
                    sample_id=self.sample_id,
                    zmq_consumer_mode="spotfinder", grid_scan_type=self.grid_scan_type
        )
        if environ["BL_ACTIVE"].lower() == "true":

            if grid.number_of_columns >= 2:
                scan_response = yield from md3_grid_scan(
                    detector=self.detector,
                    grid_width=grid.width,
                    grid_height=grid.height,
                    number_of_columns=grid.number_of_columns,
                    number_of_rows=grid.number_of_rows,
                    start_omega=self.omega.position,
                    omega_range=self.omega_range,
                    start_alignment_y=grid.initial_pos_alignment_y,
                    start_alignment_z=self.centered_loop_coordinates.alignment_z,
                    start_sample_x=grid.final_pos_sample_x,
                    start_sample_y=grid.final_pos_sample_y,
                    exposure_time=self.exposure_time,
                    user_data=user_data
                )
            else:
                scan_response = yield from md3_4d_scan(
                    detector=self.detector,
                    start_angle=self.omega.position,
                    scan_range=self.omega_range,
                    exposure_time=self.exposure_time,
                    start_alignment_y=grid.initial_pos_alignment_y,
                    stop_alignment_y=grid.final_pos_alignment_y,
                    start_sample_x=grid.center_pos_sample_x,
                    stop_sample_x=grid.center_pos_sample_x,
                    start_sample_y=grid.center_pos_sample_y,
                    stop_sample_y=grid.center_pos_sample_y,
                    start_alignment_z=self.centered_loop_coordinates.alignment_z,
                    stop_alignment_z=self.centered_loop_coordinates.alignment_z,
                    number_of_frames=grid.number_of_rows,
                    user_data=user_data
                )
        elif environ["BL_ACTIVE"].lower() == "false":
            # Trigger the simulated simplon api, and return
            # a random MD3ScanResponse
            detector_configuration = {"nimages": grid.number_of_columns*grid.number_of_rows}
            yield from arm_trigger_and_disarm_detector(
                detector=self.detector,
                detector_configuration=detector_configuration,
                metadata={},
            )
            scan_response = MD3ScanResponse(
                task_name="Raster Scan",
                task_flags=8,
                start_time="2023-02-21 12:40:47.502",
                end_time="2023-02-21 12:40:52.814",
                task_output="org.embl.dev.pmac.PmacDiagnosticInfo@64ba4055",
                task_exception="null",
                result_id=1,
            )

        self.md3_scan_response.put(scan_response.dict())
        if draw_grid_in_mxcube:
            loop = asyncio.get_event_loop()
            loop.create_task(
                self.draw_grid_in_mxcube(
                    rectangle_coordinates_in_pixels,
                    grid.number_of_columns,
                    grid.number_of_rows,
                )
            )


    def find_crystal_positions(
        self,
        sample_id: str,
        grid_scan_type: str,
        raster_grid_coordinates: RasterGridMotorCoordinates,
        filename: str = "crystal_finder_results",
    ) -> tuple[list[CrystalPositions], list[dict[str, int]], npt.NDArray, bytes]:
        """
        Finds the crystal position based on the number of spots obtained from a
        grid_scan using the CrystalFinder class. The number of spots are obtained
        from redis streams, which are generated by the mx - spotfinder in the
        mx - zmq - consumer. The order of the number of spots is assumed to
        correspond to an md3_grid_scan with the parameters:
            invert_direction = True,
            use_fast_mesh_scans = True
        TODO: Account for all md3 grid scan options

        Parameters
        ----------
        sample_id: str
            Sample id
        grid_scan_type : str
            The grid scan type. Can be either `flat` or `edge`
        raster_grid_coordinates : RasterGridMotorCoordinates
            RasterGridMotorCoordinates pydantic model
        filename: str, optional
            The name of the file used to save the CrystalFinder results if
            self.plot = True, by default crystal_finder_results

        Returns
        -------
        tuple[list[CrystalPositions], list[dict[str, int]], npt.NDArray, bytes]
            1) A list of CrystalPositions pydantic models,
            2) A list of dictionaries describing the distance between all overlapping crystals,
            3) A numpy array containing the numbers of spots.
        """
        n_rows = raster_grid_coordinates.number_of_rows
        n_cols = raster_grid_coordinates.number_of_columns
        result = []
        number_of_spots_list = []

        number_of_frames = self.redis_connection.xlen(
            f"spotfinder_results_{grid_scan_type}:{sample_id}"
        )
        while number_of_frames < n_rows * n_cols:
            # TODO: include a timeout, and notify the user that we lost frames
            # somewhere
            time.sleep(0.2)
            number_of_frames = self.redis_connection.xlen(
                f"spotfinder_results_{grid_scan_type}:{sample_id}"
            )
            logger.info(
                f"Expecting {n_rows*n_cols} frames, got {number_of_frames} frames so far"
            )

        last_id = 0
        for _ in range(number_of_frames):
            try:
                spotfinder_results, last_id = self.get_spotfinder_results(
                    sample_id, grid_scan_type, last_id
                )
                result.append(spotfinder_results)
                number_of_spots_list.append(spotfinder_results.number_of_spots)
            except IndexError:
                pass

        # spotfinder_results_array = np.array(result).reshape(n_rows, n_cols)

        # Reorder array
        number_of_spots_array = (
            np.array(number_of_spots_list).reshape(n_cols, n_rows).transpose()
        )

        crystal_finder = CrystalFinder(
            number_of_spots_array,
            threshold=self.threshold,
            grid_scan_motor_coordinates=raster_grid_coordinates,
        )
        (
            crystal_locations,
            distance_between_crystals,
        ) = crystal_finder.plot_crystal_finder_results(
            save=self.plot, filename=filename
        )

        return (
            crystal_locations,
            distance_between_crystals,
            number_of_spots_array,
        )

    def get_spotfinder_results(
        self, sample_id: str, grid_scan_type: str, id: Union[bytes, int]
    ) -> tuple[SpotfinderResults, bytes]:
        """
        Gets the spotfinder results from redis streams. The spotfinder results
        are calculated by the mx-hdf5-builder service. The name of the redis key
        follows the format f"spotfinder_results:{sample_id}"

        Parameters
        ----------
        sample_id : str
            The sample_id, e.g. my_sample
        grid_scan_type : str
            The grid scan type. Can be either `flat` or `edge`
        id: Union[bytes, int]
            id of the topic in bytes or int format

        Returns
        -------
        spotfinder_results, last_id: tuple[SpotfinderResults, bytes]
            A tuple containing SpotfinderResults and the redis streams
            last_id
        """
        topic = f"spotfinder_results_{grid_scan_type}:{sample_id}"
        response = self.redis_connection.xread({topic: id}, count=1)

        # Extract key and messages from the response
        _, messages = response[0]

        # Update last_id and store messages data
        last_id, data = messages[0]

        bluesky_event_doc = pickle.loads(data[b"bluesky_event_doc"])

        spotfinder_results = SpotfinderResults(
            type=data[b"type"],
            number_of_spots=data[b"number_of_spots"],
            image_id=data[b"image_id"],
            series_id=data[b"series_id"],
            sample_id=data[b"sample_id"],
            bluesky_event_doc=bluesky_event_doc,
            grid_scan_type=data[b"grid_scan_type"],
        )

        assert (
            sample_id == spotfinder_results.sample_id
        ), "The spotfinder sample_id is different from the queueserver sample_id"
        return spotfinder_results, last_id

    async def draw_grid_in_mxcube(
        self,
        rectangle_coordinates: dict,
        num_cols: int,
        num_rows: int,
        number_of_spots_array: Optional[npt.NDArray] = None,
    ):
        """Draws a grid in mxcube

        Parameters
        ----------
        rectangle_coordinates: dict
            Rectangle coordinates of the grid obtained from the PSI loop centering code
        num_cols: int
            Number of columns of the grid
        num_rows: int
            Number of rows of the grid
        number_of_spots_array: npt.NDArray, optional
            A numpy array of shape(n_rows, n_cols) containing the
            number of spots of the grid, by default None

        Returns
        -------
        None
        """
        self.grid_id += 1

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

        mm_per_pixel = 1 / self.zoom.pixels_per_mm
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
                    "id": f"G{self.grid_id}",
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
                    "name": f"Grid-{self.grid_id}",
                    "numCols": num_cols,
                    "numRows": num_rows,
                    "result": heatmap_and_crystalmap,
                    "screenCoord": rectangle_coordinates["top_left"].tolist(),
                    "selected": True,
                    "state": "SAVED",
                    "t": "G",
                    "pixelsPerMm": [self.zoom.pixels_per_mm, self.zoom.pixels_per_mm],
                    # 'dxMm': 1/292.8705182537115,
                    # 'dyMm': 1/292.8705182537115
                }
            ]
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    os.path.join(
                        self.mxcube_url,
                        "mxcube/api/v0.1/sampleview/shapes/create_grid",
                    ),
                    json=mxcube_payload,
                )
                logger.info(f"MXCuBE request response: {response.text}")

        except httpx.ConnectError:
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
        num_cols: int
            Number of columns
        num_rows: int
            Number of rows
        number_of_spots_array: npt.NDArray
            A numpy array of shape(n_rows, n_cols) containing the
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



def xray_centering(
    sample_id: str,
    detector: DectrisDetector,
    omega: Union[CosylabMotor, MD3Motor],
    zoom: MD3Zoom,
    grid_scan_type: str,
    exposure_time: float = 1.0,
    omega_range: float = 0.0,
    count_time: float = None
) -> Generator[Msg, None, None]:
    """
    This is a wrapper to execute the optical and xray centering plan
    using the OpticalAndXRayCentering class. This function is needed because the
    bluesky - queueserver does not interact nicely with classes.
    The default parameters used in the plan are loaded from the
    optical_and_xray_centering.yml file located in the mx3 - beamline - library
    configuration folder.

    Parameters
    ----------
    sample_id: str
        Sample id
    detector: DectrisDetector
        The dectris detector ophyd device
    omega : Union[CosylabMotor, MD3Motor]
        Omega
    zoom : MD3Zoom
        Zoom
    grid_scan_type: str
        Grid scan type, could be either `flat`, or `edge`.
    exposure_time : float
        Detector exposure time
    omega_range : float, optional
        Omega range (degrees) for the scan, by default 0
    count_time : float
        Detector count time, by default None. If this parameter is not set, 
        it is set to frame_time - 0.0000001 by default. This calculation 
        is done via the DetectorConfiguration pydantic model.

    Returns
    -------
    Generator[Msg, None, None]
    """
    _xray_centering = XRayCentering(
        sample_id=sample_id,
        detector=detector,
        omega=omega,
        zoom=zoom,
        grid_scan_type=grid_scan_type,
        exposure_time=exposure_time,
        omega_range=omega_range,
        count_time=count_time
    )
    # NOTE: We could also use the plan_stubs open_run, close_run, monitor
    # instead of `monitor_during_wrapper` and `run_wrapper` methods below
    yield from monitor_during_wrapper(
        run_wrapper(_xray_centering.start_grid_scan(), md={"sample_id": sample_id}),
        signals=(
            omega,
            zoom,
            _xray_centering.md3_scan_response,
        ),
    )
