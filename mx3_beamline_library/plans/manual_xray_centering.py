from typing import Generator

import numpy as np
from bluesky.plan_stubs import mv
from bluesky.preprocessors import monitor_during_wrapper, run_wrapper
from bluesky.utils import Msg
import pickle

from ..devices.motors import md3
from ..logger import setup_logger
from ..plans.plan_stubs import md3_move
from ..schemas.crystal_finder import MotorCoordinates
from ..schemas.loop_edge_detection import RectangleCoordinates
from ..schemas.xray_centering import RasterGridCoordinates
from .stubs.devices import validate_raster_grid_limits
from .xray_centering import XRayCentering
from ..config import redis_connection

logger = setup_logger()


class ManualXRayCentering(XRayCentering):
    """
    This plan is used to run a grid scan based on parameters
    obtained from mxcube
    """

    def __init__(
        self,
        sample_id: str,
        grid_scan_id: str | int,
        grid_top_left_coordinate: tuple[int, int] | list[int],
        grid_width: int,
        grid_height: int,
        beam_position: tuple[int, int] | list[int],
        number_of_columns: int,
        number_of_rows: int,
        detector_distance: float,
        photon_energy: float,
        transmission: float,
        md3_alignment_y_speed: float = 10.0,
        omega_range: float = 0,
        count_time: float | None = None,
        hardware_trigger=True,
    ) -> None:
        """
        Parameters
        ----------
        sample_id: str
            Sample id
        grid_scan_id: str
            Grid scan type
        grid_top_left_coordinate : Union[list, tuple[int, int]]
            Top left coordinate of the scan in pixels
        grid_width : int
            Grid width in pixels
        grid_height : int
            Grid height in pixels
        beam_position : Union[tuple[int, int], list[int]]
            Beam position in pixels
        number_of_columns : int
            Number of columns
        number_of_rows : int
            Number of rows
        detector_distance : float
            Detector distance in meters
        photon_energy : float
            Photon energy in keV
        transmission: float
            The transmission, must be a value between 0 and 1.
        md3_alignment_y_speed : float, optional
            The md3 alignment y speed measured in mm/s, by default 10.0 mm/s
        omega_range : float, optional
            Omega range (degrees) for the scan, by default 0
        count_time : float | None
            Detector count time, by default None. If this parameter is not set,
            it is set to frame_time - 0.0000001 by default. This calculation
            is done via the DetectorConfiguration pydantic model.
        hardware_trigger : bool, optional
            If set to true, we trigger the detector via hardware trigger, by default True.
            Warning! hardware_trigger=False is used mainly for debugging purposes,
            as it results in a very slow scan

        Returns
        -------
        None
        """
        super().__init__(
            sample_id=sample_id,
            grid_scan_id=grid_scan_id,
            detector_distance=detector_distance,
            photon_energy=photon_energy,
            transmission=transmission,
            omega_range=omega_range,
            md3_alignment_y_speed=md3_alignment_y_speed,
            count_time=count_time,
            hardware_trigger=hardware_trigger,
        )
        self.grid_top_left_coordinate = np.array(grid_top_left_coordinate)
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.beam_position = beam_position
        self.number_of_columns = number_of_columns
        self.number_of_rows = number_of_rows
        self.zoom = md3.zoom

    def get_optical_centering_results(self):
        """
        Override parent class
        """
        return

    def _get_current_motor_positions(self) -> MotorCoordinates:
        """Gets the current motor positions

        Returns
        -------
        MotorCoordinates
            The current motor positions
        """
        return MotorCoordinates(
            sample_x=md3.sample_x.position,
            sample_y=md3.sample_y.position,
            alignment_x=md3.alignment_x.position,
            alignment_y=md3.alignment_y.position,
            alignment_z=md3.alignment_z.position,
            omega=md3.omega.position,
        )

    def _start_grid_scan(self) -> Generator[Msg, None, None]:
        """
        Runs an edge or flat grid scan, depending on the value of self.grid_scan_id

        Yields
        ------
        Generator[Msg, None, None]
            A bluesky plan tha centers the a sample using optical and X-ray centering
        """
        md3.save_centring_position()

        if md3.phase.get() != "DataCollection":
            yield from mv(md3.phase, "DataCollection")

        initial_position = self._get_current_motor_positions()

        grid = self.prepare_raster_grid(md3.omega.position)

        validate_raster_grid_limits(grid)

        redis_connection.set(
            f"mxcube_raster_grid:sample_id_{self.sample_id}:grid_scan_id_{self.grid_scan_id}", 
            pickle.dumps(grid.model_dump())
        )


        logger.info(f"Running grid scan: {self.grid_scan_id}")

        self.md3_exposure_time = self._calculate_md3_exposure_time(grid)

        yield from self._grid_scan(grid)

        if not self.hardware_trigger:
            yield from md3_move(
                md3.sample_x,
                initial_position.sample_x,
                md3.sample_y,
                initial_position.sample_y,
                md3.alignment_x,
                initial_position.alignment_x,
                md3.alignment_y,
                initial_position.alignment_y,
                md3.alignment_z,
                initial_position.alignment_z,
                md3.omega,
                initial_position.omega,
            )
            return

        # Move the motors to the top left bottom coordinate position
        # to ensure the grid is shown correctly in mxcube
        yield from md3_move(
            md3.sample_x,
            grid.initial_pos_sample_x,
            md3.sample_y,
            grid.initial_pos_sample_y,
            md3.alignment_y,
            grid.initial_pos_alignment_y,
        )

    def start_grid_scan(self) -> Generator[Msg, None, None]:
        """
        Opens and closes the run while keeping track of the signals
        used in the manual x-ray centering plan

        Yields
        ------
        Generator[Msg, None, None]
            The plan generator
        """
        yield from monitor_during_wrapper(
            run_wrapper(self._start_grid_scan(), md={"sample_id": self.sample_id}),
            signals=(self.md3_scan_response,),
        )

    def prepare_raster_grid(self, omega: float) -> RasterGridCoordinates:
        """
        Prepares a raster grid. The limits of the grid are obtained using
        the LoopEdgeDetection class

        Parameters
        ----------
        omega : float
            Angle at which the grid scan is done

        Returns
        -------
        motor_coordinates: RasterGridCoordinates
            A pydantic model containing the initial and final motor positions of the grid,
            as well as its coordinates in units of pixels
        """
        bottom_right_coords = np.array(
            [
                self.grid_top_left_coordinate[0] + self.grid_width,
                self.grid_top_left_coordinate[1] + self.grid_height,
            ]
        )

        rectangle_coordinates = RectangleCoordinates(
            top_left=self.grid_top_left_coordinate, bottom_right=bottom_right_coords
        )

        width_pixels = abs(
            rectangle_coordinates.top_left[0] - rectangle_coordinates.bottom_right[0]
        )
        width_mm = width_pixels / self.zoom.pixels_per_mm

        height_pixels = abs(
            rectangle_coordinates.top_left[1] - rectangle_coordinates.bottom_right[1]
        )
        height_mm = height_pixels / self.zoom.pixels_per_mm

        # Y pixel coordinates
        initial_pos_y_pixels = rectangle_coordinates.top_left[1] - self.beam_position[1]
        final_pos_y_pixels = (
            rectangle_coordinates.bottom_right[1] - self.beam_position[1]
        )

        # Alignment y target positions (mm)
        initial_pos_alignment_y = (
            md3.alignment_y.position + initial_pos_y_pixels / self.zoom.pixels_per_mm
        )
        final_pos_alignment_y = (
            md3.alignment_y.position + final_pos_y_pixels / self.zoom.pixels_per_mm
        )

        # X pixel coordinates
        initial_pos_x_pixels = rectangle_coordinates.top_left[0] - self.beam_position[0]
        final_pos_x_pixels = (
            rectangle_coordinates.bottom_right[0] - self.beam_position[0]
        )

        # Sample x target positions (mm)
        initial_pos_sample_x = md3.sample_x.position + np.sin(
            np.radians(md3.omega.position)
        ) * (initial_pos_x_pixels / self.zoom.pixels_per_mm)
        final_pos_sample_x = md3.sample_x.position + np.sin(
            np.radians(md3.omega.position)
        ) * (+final_pos_x_pixels / self.zoom.pixels_per_mm)

        # Sample y target positions (mm)
        initial_pos_sample_y = md3.sample_y.position + np.cos(
            np.radians(md3.omega.position)
        ) * (initial_pos_x_pixels / self.zoom.pixels_per_mm)
        final_pos_sample_y = md3.sample_y.position + np.cos(
            np.radians(md3.omega.position)
        ) * (final_pos_x_pixels / self.zoom.pixels_per_mm)

        # Center of the grid (mm) (y-axis only)
        center_x_of_grid_pixels = (
            rectangle_coordinates.top_left[0] + rectangle_coordinates.bottom_right[0]
        ) / 2
        center_pos_sample_x = md3.sample_x.position + np.sin(
            np.radians(md3.omega.position)
        ) * (
            (center_x_of_grid_pixels - self.beam_position[0]) / self.zoom.pixels_per_mm
        )
        center_pos_sample_y = md3.sample_y.position + np.cos(
            np.radians(md3.omega.position)
        ) * (
            (center_x_of_grid_pixels - self.beam_position[0]) / self.zoom.pixels_per_mm
        )

        # NOTE: The width and height are measured in mm and the grid_step in micrometers,
        # hence the conversion below

        raster_grid_coordinates = RasterGridCoordinates(
            use_centring_table=True,
            initial_pos_sample_x=initial_pos_sample_x,
            final_pos_sample_x=final_pos_sample_x,
            initial_pos_sample_y=initial_pos_sample_y,
            final_pos_sample_y=final_pos_sample_y,
            initial_pos_alignment_y=initial_pos_alignment_y,
            final_pos_alignment_y=final_pos_alignment_y,
            initial_pos_alignment_z=md3.alignment_z.position,
            final_pos_alignment_z=md3.alignment_z.position,
            omega=omega,
            alignment_x_pos=md3.alignment_x.position,
            width_mm=width_mm,
            height_mm=height_mm,
            center_pos_sample_x=center_pos_sample_x,
            center_pos_sample_y=center_pos_sample_y,
            number_of_columns=self.number_of_columns,
            number_of_rows=self.number_of_rows,
            top_left_pixel_coordinates=tuple(rectangle_coordinates.top_left),
            bottom_right_pixel_coordinates=tuple(rectangle_coordinates.bottom_right),
            width_pixels=width_pixels,
            height_pixels=height_pixels,
            pixels_per_mm=self.zoom.pixels_per_mm,
        )

        return raster_grid_coordinates
