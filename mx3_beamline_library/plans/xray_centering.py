import asyncio
import logging
import os
import pickle
from os import environ
from typing import Generator, Optional, Union

import httpx
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import redis
from bluesky.plan_stubs import mv
from bluesky.preprocessors import monitor_during_wrapper, run_wrapper
from bluesky.utils import Msg
from ophyd import Signal

from ..devices.classes.detectors import DectrisDetector
from ..devices.classes.motors import CosylabMotor, MD3Motor, MD3Zoom
from ..devices.motors import md3
from ..plans.basic_scans import md3_4d_scan, md3_grid_scan, slow_grid_scan
from ..schemas.crystal_finder import MotorCoordinates
from ..schemas.detector import UserData
from ..schemas.optical_centering import CenteredLoopMotorCoordinates
from ..schemas.xray_centering import RasterGridCoordinates
from .plan_stubs import md3_move

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
        grid_scan_id: str,
        exposure_time: float = 0.002,
        omega_range: float = 0.0,
        count_time: float = None,
        hardware_trigger=True,
        detector_distance: float = -0.298,
        photon_energy: float = 12700,
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
        grid_scan_id: str
            Grid scan type, could be either `flat`, or `edge`.
        exposure_time : float
            Detector exposure time (also know as frame time). NOTE: This is NOT the
            exposure time as defined by the MD3.
        omega_range : float, optional
            Omega range (degrees) for the scan, by default 0
        count_time : float
            Detector count time, by default None. If this parameter is not set,
            it is set to frame_time - 0.0000001 by default. This calculation
            is done via the DetectorConfiguration pydantic model.
        hardware_trigger : bool, optional
            If set to true, we trigger the detector via hardware trigger, by default True.
            Warning! hardware_trigger=False is used mainly for debugging purposes,
            as it results in a very slow scan
        detector_distance: float, optional
            The detector distance, by default -0.298
        photon_energy: float, optional
            The photon energy in eV, by default 12700

        Returns
        -------
        None
        """
        self.sample_id = sample_id
        self.detector = detector
        self.omega = omega
        self.zoom = zoom
        self.grid_scan_id = grid_scan_id
        self.exposure_time = exposure_time
        self.omega_range = omega_range
        self.count_time = count_time
        self.hardware_trigger = hardware_trigger
        self.detector_distance = detector_distance
        self.photon_energy = photon_energy

        self.maximum_motor_y_speed = 14.8  # mm/s

        REDIS_HOST = environ.get("REDIS_HOST", "0.0.0.0")
        REDIS_PORT = int(environ.get("REDIS_PORT", "6379"))
        self.redis_connection = redis.StrictRedis(
            host=REDIS_HOST, port=REDIS_PORT, db=0
        )

        self.mxcube_url = environ.get("MXCUBE_URL", "http://localhost:8090")

        self.grid_id = 0

        self.md3_scan_response = Signal(name="md3_scan_response", kind="normal")
        self.centered_loop_coordinates = None
        self.get_optical_centering_results()

    def get_optical_centering_results(self):
        results = pickle.loads(
            self.redis_connection.get(f"optical_centering_results:{self.sample_id}")
        )
        if not results["optical_centering_successful"]:
            raise ValueError(
                "Optical centering was not successful, grid scan cannot be executed"
            )
        self.centered_loop_coordinates = CenteredLoopMotorCoordinates.parse_obj(
            results["centered_loop_coordinates"]
        )
        self.edge_angle = results["edge_angle"]
        self.flat_angle = results["flat_angle"]
        self.flat_grid_motor_coordinates = RasterGridCoordinates.parse_obj(
            results["flat_grid_motor_coordinates"]
        )
        self.edge_grid_motor_coordinates = RasterGridCoordinates.parse_obj(
            results["edge_grid_motor_coordinates"]
        )

    def start_grid_scan(self) -> Generator[Msg, None, None]:
        """
        Runs an edge or flat grid scan, depending on the value of self.grid_scan_id

        Yields
        ------
        Generator[Msg, None, None]
            A bluesky plan tha centers the a sample using optical and X-ray centering
        """

        if self.grid_scan_id.lower() == "flat":
            grid = self.flat_grid_motor_coordinates
            yield from mv(self.omega, self.flat_angle)
        elif self.grid_scan_id.lower() == "edge":
            yield from mv(self.omega, self.edge_angle)
            grid = self.edge_grid_motor_coordinates

        logger.info(f"Running grid scan: {self.grid_scan_id}")
        self.md3_exposure_time = grid.number_of_rows * self.exposure_time

        speed_alignment_y = grid.height_mm / self.md3_exposure_time
        logger.info(f"MD3 alignment y speed: {speed_alignment_y}")

        if speed_alignment_y > self.maximum_motor_y_speed:
            raise ValueError(
                "The grid scan exceeds the maximum speed of the alignment y motor "
                f"({self.maximum_motor_y_speed} mm/s). "
                f"The current speed is {speed_alignment_y} mm/s. "
                "Increase the exposure time"
            )

        yield from self._grid_scan(grid)

    def _grid_scan(
        self,
        grid: RasterGridCoordinates,
        draw_grid_in_mxcube: bool = False,
        rectangle_coordinates_in_pixels: dict = None,
    ) -> None:
        """
        Runs an md3_grid_scan or md3_4d_scan depending on the number of rows an columns

        Parameters
        ----------
        grid : RasterGridCoordinates
            A RasterGridCoordinates object which contains information about the
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
        logger.info(f"Grid width [mm]: {grid.width_mm}")
        logger.info(f"Grid height [mm]: {grid.height_mm}")

        # NOTE: The md3_grid_scan does not like number_of_columns < 2. If
        # number_of_columns < 2 we use the md3_3d_scan instead, setting scan_range=0,
        # and keeping the values of sample_x, sample_y, and alignment_z constant
        user_data = UserData(
            id=self.sample_id,
            zmq_consumer_mode="spotfinder",
            grid_scan_id=self.grid_scan_id,
            number_of_columns=grid.number_of_columns,
            number_of_rows=grid.number_of_rows,
        )
        if self.grid_scan_id.lower() == "flat":
            start_omega = self.flat_angle
        elif self.grid_scan_id.lower() == "edge":
            start_omega = self.edge_angle
        else:
            start_omega = self.omega.position

        if environ["BL_ACTIVE"].lower() == "true":
            if self.hardware_trigger:
                if self.centered_loop_coordinates is not None:
                    start_alignment_z = self.centered_loop_coordinates.alignment_z
                else:
                    start_alignment_z = md3.alignment_z.position

                if grid.number_of_columns >= 2:
                    scan_response = yield from md3_grid_scan(
                        detector=self.detector,
                        grid_width=grid.width_mm,
                        grid_height=grid.height_mm,
                        number_of_columns=grid.number_of_columns,
                        number_of_rows=grid.number_of_rows,
                        start_omega=start_omega,
                        omega_range=self.omega_range,
                        start_alignment_y=grid.initial_pos_alignment_y,
                        start_alignment_z=start_alignment_z,
                        start_sample_x=grid.final_pos_sample_x,
                        start_sample_y=grid.final_pos_sample_y,
                        md3_exposure_time=self.md3_exposure_time,
                        user_data=user_data,
                        count_time=self.count_time,
                        detector_distance=self.detector_distance,
                        photon_energy=self.photon_energy,
                    )
                else:
                    # When we run an md3 4D scan, the md3 does not
                    # go back to the initial position, whereas when
                    # we run an md3 grid scan it does. For this reason,
                    # when we execute a 4D scan,
                    # we manually move the motors back to the initial
                    # position when the scan is finished. This is especially
                    # relevant for manual data collection
                    initial_positions = MotorCoordinates(
                        sample_x=md3.sample_x.position,
                        sample_y=md3.sample_y.position,
                        alignment_x=md3.alignment_x.position,
                        alignment_y=md3.alignment_y.position,
                        alignment_z=md3.alignment_z.position,
                        omega=md3.omega.position,
                    )
                    scan_response = yield from md3_4d_scan(
                        detector=self.detector,
                        start_angle=start_omega,
                        scan_range=self.omega_range,
                        md3_exposure_time=self.md3_exposure_time,
                        start_alignment_y=grid.initial_pos_alignment_y,
                        stop_alignment_y=grid.final_pos_alignment_y,
                        start_sample_x=grid.center_pos_sample_x,
                        stop_sample_x=grid.center_pos_sample_x,
                        start_sample_y=grid.center_pos_sample_y,
                        stop_sample_y=grid.center_pos_sample_y,
                        start_alignment_z=start_alignment_z,
                        stop_alignment_z=start_alignment_z,
                        number_of_frames=grid.number_of_rows,
                        user_data=user_data,
                        count_time=self.count_time,
                        detector_distance=self.detector_distance,
                        photon_energy=self.photon_energy,
                    )
                    yield from md3_move(
                        md3.sample_x,
                        initial_positions.sample_x,
                        md3.sample_y,
                        initial_positions.sample_y,
                        md3.alignment_x,
                        initial_positions.alignment_x,
                        md3.alignment_y,
                        initial_positions.alignment_y,
                        md3.alignment_z,
                        initial_positions.alignment_z,
                        md3.omega,
                        initial_positions.omega,
                    )
            else:
                detector_configuration = {
                    "nimages": 1,
                    "user_data": user_data.dict(),
                    "trigger_mode": "ints",
                    "ntrigger": grid.number_of_columns * grid.number_of_rows,
                }

                scan_response = yield from slow_grid_scan(
                    raster_grid_coords=grid,
                    detector=self.detector,
                    detector_configuration=detector_configuration,
                    alignment_y=md3.alignment_y,
                    alignment_z=md3.alignment_z,
                    sample_x=md3.sample_x,
                    sample_y=md3.sample_y,
                    omega=md3.omega,
                    use_centring_table=True,
                )
        elif environ["BL_ACTIVE"].lower() == "false":
            detector_configuration = {
                "nimages": 1,
                "user_data": user_data.dict(),
                "trigger_mode": "ints",
                "ntrigger": grid.number_of_columns * grid.number_of_rows,
            }

            scan_response = yield from slow_grid_scan(
                raster_grid_coords=grid,
                detector=self.detector,
                detector_configuration=detector_configuration,
                alignment_y=md3.alignment_y,
                alignment_z=md3.alignment_z,
                sample_x=md3.sample_x,
                sample_y=md3.sample_y,
                omega=md3.omega,
                use_centring_table=True,
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
            Rectangle coordinates of the grid obtained from optical centering plan
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
    grid_scan_id: str,
    exposure_time: float = 1.0,
    omega_range: float = 0.0,
    count_time: float = None,
    hardware_trigger: bool = True,
    detector_distance: float = 0.298,
    photon_energy: float = 12700,
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
    grid_scan_id: str
        Grid scan type, could be either `flat`, or `edge`.
    exposure_time : float
        Detector exposure time
    omega_range : float, optional
        Omega range (degrees) for the scan, by default 0
    count_time : float
        Detector count time, by default None. If this parameter is not set,
        it is set to frame_time - 0.0000001 by default. This calculation
        is done via the DetectorConfiguration pydantic model.
    detector_distance: float, optional
        The detector distance, by default -0.298
    photon_energy: float, optional
        The photon energy in eV, by default 12700

    Returns
    -------
    Generator[Msg, None, None]
    """
    _xray_centering = XRayCentering(
        sample_id=sample_id,
        detector=detector,
        omega=omega,
        zoom=zoom,
        grid_scan_id=grid_scan_id,
        exposure_time=exposure_time,
        omega_range=omega_range,
        count_time=count_time,
        hardware_trigger=hardware_trigger,
        detector_distance=detector_distance,
        photon_energy=photon_energy,
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
