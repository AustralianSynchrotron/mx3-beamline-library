from typing import Union, Generator
from mx3_beamline_library.devices.classes.detectors import DectrisDetector
from mx3_beamline_library.devices.classes.motors import CosylabMotor, MD3Motor, MD3Zoom
from .xray_centering import XRayCentering
from ..devices.motors import md3
from ..schemas.xray_centering import RasterGridCoordinates
from ..schemas.loop_edge_detection import RectangleCoordinates
import logging
from bluesky.plan_stubs import mv
from bluesky.utils import Msg
import numpy as np

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)


class ManualXRayCentering(XRayCentering):
    def __init__(
            self, 
            sample_id: str, 
            detector: DectrisDetector, 
            omega: Union[CosylabMotor, MD3Motor], 
            zoom: MD3Zoom, 
            grid_scan_id: str,
            grid_top_left_coordinate: Union[list, tuple[int, int]], 
            grid_width: str,
            grid_height: str,
            beam_position: Union[tuple[int, int], list[int]],
            number_of_columns: int,
            number_of_rows: int,
            exposure_time: float = 0.002, 
            omega_range: float = 0, 
            count_time: float = None, 
            hardware_trigger=True
            ) -> None:
        super().__init__(
            sample_id, 
            detector, 
            omega, 
            zoom, 
            grid_scan_id, 
            exposure_time, 
            omega_range, 
            count_time, 
            hardware_trigger
        )
        self.grid_top_left_coordinate = np.array(grid_top_left_coordinate)
        self.grid_width = grid_width
        self.grid_height = grid_height 
        self.beam_position = beam_position
        self.number_of_columns = number_of_columns
        self.number_of_rows = number_of_rows


    def get_optical_centering_results(self):
        """
        Override parent class
        """
        return

    def start_grid_scan(self) -> Generator[Msg, None, None]:
        """
        Runs an edge or flat grid scan, depending on the value of self.grid_scan_id

        Yields
        ------
        Generator[Msg, None, None]
            A bluesky plan tha centers the a sample using optical and X-ray centering
        """
        if md3.phase.get() != "DataCollection":
            yield from mv(md3.phase, "DataCollection")

        grid = self.prepare_raster_grid(md3.omega.position)

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


    def prepare_raster_grid(
        self, omega: float
    ) -> RasterGridCoordinates:
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
        rectangle_coordinates: dict
            Rectangle coordinates in pixels
        """
        bottom_right_coords = np.array(
            [self.grid_top_left_coordinate[0] + self.grid_width, 
             self.grid_top_left_coordinate[1] + self.grid_height
             ])


        rectangle_coordinates = RectangleCoordinates(
            top_left=self.grid_top_left_coordinate,
            bottom_right=bottom_right_coords
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
        initial_pos_y_pixels = abs(
            rectangle_coordinates.top_left[1] - self.beam_position[1]
        )
        final_pos_y_pixels = abs(
            rectangle_coordinates.bottom_right[1] - self.beam_position[1]
        )

        # Alignment y target positions (mm)
        initial_pos_alignment_y = (
            md3.alignment_y.position - initial_pos_y_pixels / self.zoom.pixels_per_mm
        )
        final_pos_alignment_y = (
            md3.alignment_y.position + final_pos_y_pixels / self.zoom.pixels_per_mm
        )

        # X pixel coordinates
        initial_pos_x_pixels = abs(
            rectangle_coordinates.top_left[0] - self.beam_position[0]
        )
        final_pos_x_pixels = abs(
            rectangle_coordinates.bottom_right[0] - self.beam_position[0]
        )

        # Sample x target positions (mm)
        initial_pos_sample_x = md3.sample_x.position - np.sin(
            np.radians(self.omega.position)
        ) * (initial_pos_x_pixels / self.zoom.pixels_per_mm)
        final_pos_sample_x = md3.sample_x.position + np.sin(
            np.radians(self.omega.position)
        ) * (+final_pos_x_pixels / self.zoom.pixels_per_mm)

        # Sample y target positions (mm)
        initial_pos_sample_y = md3.sample_y.position - np.cos(
            np.radians(self.omega.position)
        ) * (initial_pos_x_pixels / self.zoom.pixels_per_mm)
        final_pos_sample_y = md3.sample_y.position + np.cos(
            np.radians(self.omega.position)
        ) * (final_pos_x_pixels / self.zoom.pixels_per_mm)

        # Center of the grid (mm) (y-axis only)
        center_x_of_grid_pixels = (
            rectangle_coordinates.top_left[0] + rectangle_coordinates.bottom_right[0]
        ) / 2
        center_pos_sample_x = md3.sample_x.position + np.sin(
            np.radians(self.omega.position)
        ) * (
            (center_x_of_grid_pixels - self.beam_position[0]) / self.zoom.pixels_per_mm
        )
        center_pos_sample_y = md3.sample_y.position + np.cos(
            np.radians(self.omega.position)
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