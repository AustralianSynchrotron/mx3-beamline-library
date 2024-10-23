import pickle
from typing import Generator

from bluesky.plan_stubs import mv
from bluesky.preprocessors import monitor_during_wrapper, run_wrapper
from bluesky.utils import Msg
from ophyd import Signal

from ..config import BL_ACTIVE, redis_connection
from ..devices.detectors import dectris_detector
from ..devices.motors import md3
from ..logger import setup_logger
from ..plans.basic_scans import md3_4d_scan, md3_grid_scan, slow_grid_scan
from ..schemas.crystal_finder import MotorCoordinates
from ..schemas.detector import UserData
from ..schemas.optical_centering import CenteredLoopMotorCoordinates
from ..schemas.xray_centering import RasterGridCoordinates
from .plan_stubs import md3_move
from .stubs.devices import validate_raster_grid_limits

logger = setup_logger()


class XRayCentering:
    """
    This plan runs a grid scan based on coordinates found by the OpticalCentering
    plan
    """

    def __init__(
        self,
        sample_id: str,
        grid_scan_id: str,
        exposure_time: float = 0.002,
        omega_range: float = 0.0,
        count_time: float = None,
        hardware_trigger=True,
        detector_distance: float = -0.298,
        photon_energy: float = 12.7,
    ) -> None:
        """
        Parameters
        ----------
        sample_id: str
            Sample id
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
            The photon energy in keV, by default 12.7

        Returns
        -------
        None
        """
        self.sample_id = sample_id
        self.grid_scan_id = grid_scan_id
        self.exposure_time = exposure_time
        self.omega_range = omega_range
        self.count_time = count_time
        self.hardware_trigger = hardware_trigger
        self.detector_distance = detector_distance
        self.photon_energy = photon_energy

        self.maximum_motor_y_speed = 14.8  # mm/s

        self.md3_scan_response = Signal(name="md3_scan_response", kind="normal")
        self.centered_loop_coordinates = None
        self.get_optical_centering_results()

    def get_optical_centering_results(self) -> None:
        """
        Gets the optical centering results from redis. This means that
        the optical centering plan has to be executed before running this plan

        Raises
        ------
        ValueError
            An error if the optical centering results are not found

        Returns
        -------
        None
        """
        key = redis_connection.get(f"optical_centering_results:{self.sample_id}")
        if key is None:
            raise ValueError(
                "Could not find optical centering results in redis for sample_id: "
                f"{self.sample_id}"
            )

        results = pickle.loads(key)

        if not results["optical_centering_successful"]:
            raise ValueError(
                "Optical centering was not successful, grid scan cannot be executed"
            )
        self.centered_loop_coordinates = CenteredLoopMotorCoordinates.model_validate(
            results["centered_loop_coordinates"]
        )
        self.edge_angle = results["edge_angle"]
        self.flat_angle = results["flat_angle"]
        self.flat_grid_motor_coordinates = RasterGridCoordinates.model_validate(
            results["flat_grid_motor_coordinates"]
        )
        self.edge_grid_motor_coordinates = RasterGridCoordinates.model_validate(
            results["edge_grid_motor_coordinates"]
        )

        validate_raster_grid_limits(self.flat_grid_motor_coordinates)

        validate_raster_grid_limits(self.edge_grid_motor_coordinates)

    def _start_grid_scan(self) -> Generator[Msg, None, None]:
        """
        Runs an edge or flat grid scan, depending on the value of self.grid_scan_id

        Yields
        ------
        Generator[Msg, None, None]
            A bluesky plan tha centers the a sample using optical and X-ray centering
        """
        if md3.phase.get() != "DataCollection":
            yield from mv(md3.phase, "DataCollection")

        if self.grid_scan_id.lower() == "flat":
            grid = self.flat_grid_motor_coordinates
            yield from mv(md3.omega, self.flat_angle)
        elif self.grid_scan_id.lower() == "edge":
            yield from mv(md3.omega, self.edge_angle)
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
    ) -> Generator[Msg, None, None]:
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
        if isinstance(self.grid_scan_id, str):
            if self.grid_scan_id.lower() == "flat":
                start_omega = self.flat_angle
            elif self.grid_scan_id.lower() == "edge":
                start_omega = self.edge_angle
            else:
                start_omega = md3.omega.position
        else:
            start_omega = md3.omega.position

        if BL_ACTIVE == "true":
            if self.hardware_trigger:
                if self.centered_loop_coordinates is not None:
                    start_alignment_z = self.centered_loop_coordinates.alignment_z
                else:
                    start_alignment_z = md3.alignment_z.position

                if grid.number_of_columns >= 2:
                    scan_response = yield from md3_grid_scan(
                        detector=dectris_detector,
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
                        detector=dectris_detector,
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
                    "user_data": user_data.model_dump(),
                    "trigger_mode": "ints",
                    "ntrigger": grid.number_of_columns * grid.number_of_rows,
                }

                scan_response = yield from slow_grid_scan(
                    raster_grid_coords=grid,
                    detector=dectris_detector,
                    detector_configuration=detector_configuration,
                    alignment_y=md3.alignment_y,
                    alignment_z=md3.alignment_z,
                    sample_x=md3.sample_x,
                    sample_y=md3.sample_y,
                    omega=md3.omega,
                    use_centring_table=True,
                )
        elif BL_ACTIVE == "false":
            detector_configuration = {
                "nimages": 1,
                "user_data": user_data.model_dump(),
                "trigger_mode": "ints",
                "ntrigger": grid.number_of_columns * grid.number_of_rows,
            }

            scan_response = yield from slow_grid_scan(
                raster_grid_coords=grid,
                detector=dectris_detector,
                detector_configuration=detector_configuration,
                alignment_y=md3.alignment_y,
                alignment_z=md3.alignment_z,
                sample_x=md3.sample_x,
                sample_y=md3.sample_y,
                omega=md3.omega,
                use_centring_table=True,
            )
        yield from mv(self.md3_scan_response, str(scan_response.model_dump()))

    def start_grid_scan(self) -> Generator[Msg, None, None]:
        """
        Opens and closes the run while keeping track of the signals
        used in the x-ray centering plan

        Yields
        ------
        Generator[Msg, None, None]
            The plan generator
        """
        yield from monitor_during_wrapper(
            run_wrapper(self._start_grid_scan(), md={"sample_id": self.sample_id}),
            signals=(
                md3.omega,
                md3.zoom,
                self.md3_scan_response,
            ),
        )
