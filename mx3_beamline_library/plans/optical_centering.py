import logging
import pickle
from os import environ, path
from typing import Generator, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import redis
import yaml
from bluesky.plan_stubs import mv
from bluesky.preprocessors import monitor_during_wrapper, run_wrapper
from bluesky.utils import Msg
from ophyd import Signal
from ophyd.signal import ConnectionTimeoutError
from scipy import optimize

from ..constants import top_camera_background_img_array
from ..devices.classes.detectors import BlackFlyCam, MDRedisCam
from ..devices.classes.motors import (
    CosylabMotor,
    MD3BackLight,
    MD3Motor,
    MD3Phase,
    MD3Zoom,
)
from ..schemas.optical_centering import (
    CenteredLoopMotorCoordinates,
    OpticalCenteringResults,
)
from ..schemas.xray_centering import RasterGridMotorCoordinates
from ..science.optical_and_loop_centering.psi_optical_centering import (
    loopImageProcessing,
)

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)


class OpticalCentering:
    """
    This class runs a bluesky plan that optically centers the loop
    using the loop finder code developed by PSI. Before analysing an image,
    we can unblur the image at the start of the plan to make sure the
    results are consistent. Finally we find angles at which the area of a
    loop is maximum and minimum (flat and edge)
    """

    def __init__(
        self,
        sample_id: str,
        md3_camera: MDRedisCam,
        top_camera: BlackFlyCam,
        sample_x: Union[CosylabMotor, MD3Motor],
        sample_y: Union[CosylabMotor, MD3Motor],
        alignment_x: Union[CosylabMotor, MD3Motor],
        alignment_y: Union[CosylabMotor, MD3Motor],
        alignment_z: Union[CosylabMotor, MD3Motor],
        omega: Union[CosylabMotor, MD3Motor],
        zoom: MD3Zoom,
        phase: MD3Phase,
        backlight: MD3BackLight,
        beam_position: tuple[int, int],
        beam_size: tuple[float, float],
        auto_focus: bool = True,
        min_focus: float = -0.3,
        max_focus: float = 1.3,
        tol: float = 0.3,
        number_of_intervals: int = 2,
        plot: bool = False,
        loop_img_processing_beamline: str = "MX3",
        loop_img_processing_zoom: str = "1",
        number_of_omega_steps: int = 7,
        x_pixel_target: int = 841,
        y_pixel_target: int = 472,
        top_camera_background_img_array: npt.NDArray = None,
        top_camera_roi_x: tuple[int, int] = (0, 1224),
        top_camera_roi_y: tuple[int, int] = (100, 1024),
    ) -> None:
        """
        Parameters
        ----------
        sample_id : str
            Sample id
        md3_camera : MDRedisCam
            MD3 Camera
        top_camera : BlackFlyCam
            Top camera
        sample_x : Union[CosylabMotor, MD3Motor]
            Sample x
        sample_y : Union[CosylabMotor, MD3Motor]
            Sample y
        alignment_x : Union[CosylabMotor, MD3Motor]
            Alignment x
        alignment_y : Union[CosylabMotor, MD3Motor]
            Alignment y
        alignment_z : Union[CosylabMotor, MD3Motor]
            Alignment y
        omega : Union[CosylabMotor, MD3Motor]
            Omega
        zoom : MD3Zoom
            Zoom
        phase : MD3Phase
            MD3 phase ophyd-signal
        backlight : MD3Backlight
            Backlight
        beam_position : tuple[int, int]
            Position of the beam
        beam_size : tuple[float, float]
            Beam size
        auto_focus : bool, optional
            If true, we autofocus the image once before running the loop centering,
            algorithm, by default True
        min_focus : float, optional
            Minimum value to search for the maximum of var( Img * L(x,y) ),
            by default 0.0
        max_focus : float, optional
            Maximum value to search for the maximum of var( Img * L(x,y) ),
            by default 1.3
        tol : float, optional
            The tolerance used by the Golden-section search, by default 0.5
        number_of_intervals : int, optional
            Number of intervals used to find local maximums of the function
            `var( Img * L(x,y) )`, by default 2
        plot : bool, optional
            If true, we take snapshots of the loop at different stages
            of the plan, by default False
        loop_img_processing_beamline : str, optional
            This name is used to get the configuration parameters used by the
            loop image processing code developed by PSI, by default testrig
        loop_img_processing_zoom : str, optional
            We get the configuration parameters used by the loop image processing code
            for a particular zoom, by default 1.0
        number_of_omega_steps : int, optional
            Number of omega steps between 0 and 180 degrees used to find the edge and flat
            surface of the loop, by default 7
        x_pixel_target : float, optional
            We use the top camera to move the loop to the md3 camera field of view.
            x_pixel_target is the pixel coordinate that corresponds
            to the position where the loop is seen fully by the md3 camera, by default 841.0
        y_pixel_target : float, optional
            We use the top camera to move the loop to the md3 camera field of view.
            y_pixel_target is the pixel coordinate that corresponds
            to the position where the loop is seen fully by the md3 camera, by default 841.0
        top_camera_background_img_array : npt.NDArray, optional
            Top camera background image array used to determine if there is a pin.
            If top_camera_background_img_array is None, we use the default background image from
            the mx3-beamline-library
        top_camera_roi_x : tuple[int, int]
            X Top camera region of interest, by default (0, 1224)
        top_camera_roi_y : tuple[int, int]
            Y Top camera region of interest, by default (100, 1024)

        Returns
        -------
        None
        """
        self.sample_id = sample_id
        self.md3_camera = md3_camera
        self.top_camera = top_camera
        self.sample_x = sample_x
        self.sample_y = sample_y
        self.alignment_x = alignment_x
        self.alignment_y = alignment_y
        self.alignment_z = alignment_z
        self.omega = omega
        self.zoom = zoom
        self.phase = phase
        self.backlight = backlight
        self.beam_position = beam_position
        self.beam_size = beam_size
        self.auto_focus = auto_focus
        self.min_focus = min_focus
        self.max_focus = max_focus
        self.tol = tol
        self.number_of_intervals = number_of_intervals
        self.plot = plot
        self.loop_img_processing_beamline = loop_img_processing_beamline
        self.loop_img_processing_zoom = loop_img_processing_zoom
        self.number_of_omega_steps = number_of_omega_steps
        self.x_pixel_target = x_pixel_target
        self.y_pixel_target = y_pixel_target

        self.centered_loop_coordinates = None

        REDIS_HOST = environ.get("REDIS_HOST", "0.0.0.0")
        REDIS_PORT = int(environ.get("REDIS_PORT", "6379"))
        self.redis_connection = redis.StrictRedis(
            host=REDIS_HOST, port=REDIS_PORT, db=0
        )

        self.grid_scan_coordinates_flat = Signal(
            name="grid_scan_coordinates_flat", kind="normal"
        )
        self.grid_scan_coordinates_edge = Signal(
            name="grid_scan_coordinates_edge", kind="normal"
        )
        if top_camera_background_img_array is None:
            self.top_camera_background_img_array = top_camera_background_img_array

        self.top_camera_roi_x = top_camera_roi_x
        self.top_camera_roi_y = top_camera_roi_y

    def center_loop(self):
        """
        This plan is the main optical loop centering plan. Here, we optically
        center the loop using the loop centering code developed by PSI. Before
        analysing an image,  we unblur the image at to make sure the results are
        consistent. After finding the centered loop positions (motor coordinates),
        we find the edge and flat angles of the loop. Finally, the results
        are saved to redis following the convention:
            f"optical_centering_results:{self.sample_id}"
        These results are meant to be used by the
        optical_and_xray_centering plan.


        Yields
        ------
        Generator[Msg, None, None]
            A plan that automatically centers a loop
        """
        # Set phase to `Centring`
        current_phase = self.phase.get()
        if current_phase != "Centring":
            yield from mv(self.phase, "Centring")

        loop_found = yield from self.move_loop_to_md3_field_of_view()
        if not loop_found:
            return

        # We center the loop at two different zooms
        zoom_list = [1, 4]
        for zoom_value in zoom_list:
            x_coords, y_coords, omega_positions = [], [], []
            yield from mv(self.zoom, zoom_value)
            omega_list = [0, 90, 180]
            for omega in omega_list:
                yield from mv(self.omega, omega)
                if self.auto_focus and zoom_value == 1:
                    yield from self.unblur_image(
                        self.alignment_x,
                        self.min_focus,
                        self.max_focus,
                        self.tol,
                        self.number_of_intervals,
                    )

                x, y = self.find_loop_edge_coordinates()
                logger.info(f"X pixel: {x}")
                logger.info(f"Y pixel: {y}")
                x_coords.append(x / self.zoom.pixels_per_mm)
                y_coords.append(y / self.zoom.pixels_per_mm)
                omega_positions.append(np.radians(self.omega.position))
            logger.info(f"Omega positions: {omega_positions}")
            yield from self.drive_motors_to_aligned_position(
                x_coords, y_coords, omega_positions
            )
            self.centered_loop_coordinates = CenteredLoopMotorCoordinates(
                alignment_x=self.alignment_x.position,
                alignment_y=self.alignment_y.position,
                alignment_z=self.alignment_z.position,
                sample_x=self.sample_x.position,
                sample_y=self.sample_y.position,
            )

        successful_centering = yield from self.find_edge_and_flat_angles()

        if not successful_centering:
            optical_centering_results = OpticalCenteringResults(
                optical_centering_successful=False
            )
            self.redis_connection.set(
                f"optical_centering_results:{self.sample_id}",
                pickle.dumps(optical_centering_results.dict()),
            )
            return

        # Step 3: Prepare raster grids for the edge surface
        yield from mv(self.zoom, 4, self.omega, self.edge_angle)
        grid_edge, _ = self.prepare_raster_grid(
            self.edge_angle, f"{self.sample_id}_raster_grid_edge"
        )
        # Add metadata for bluesky documents
        self.grid_scan_coordinates_edge.put(grid_edge.dict())

        # Step 3: Prepare raster grids for the flat surface
        yield from mv(self.zoom, 4, self.omega, self.flat_angle)
        grid_flat, rectangle_coordinates_flat = self.prepare_raster_grid(
            self.flat_angle, f"{self.sample_id}_raster_grid_flat"
        )
        # Add metadata for bluesky documents
        self.grid_scan_coordinates_flat.put(grid_flat.dict())

        optical_centering_results = OpticalCenteringResults(
            optical_centering_successful=True,
            centered_loop_coordinates=self.centered_loop_coordinates,
            edge_angle=self.edge_angle,
            flat_angle=self.flat_angle,
            edge_grid_motor_coordinates=grid_edge,
            flat_grid_motor_coordinates=grid_flat,
        )

        # Save results to redis for the
        self.redis_connection.set(
            f"optical_centering_results:{self.sample_id}",
            pickle.dumps(optical_centering_results.dict()),
        )
        logger.info("Optical centering successful!")

    def drive_motors_to_aligned_position(
        self, x_coords: list, y_coords: list, omega_positions: list
    ):
        """
        Drives motors to an aligned position based on a list of x and y coordinates
        (in units of mm), and a list of omega positions (in units of radians).

        Parameters
        ----------
        x_coords : list
            X coordinates in mm
        y_coords : list
            Y coordinates in mm
        omega_positions : list
            Omega positions in units of radians

        Yields
        ------
        Generator[Msg, None, None]
            A plan that centers a loop
        """
        chi_angle = np.radians(90)
        chiRotMatrix = np.matrix(
            [
                [np.cos(chi_angle), -np.sin(chi_angle)],
                [np.sin(chi_angle), np.cos(chi_angle)],
            ]
        )
        Z = chiRotMatrix * np.matrix([x_coords, y_coords])
        z = Z[1]
        avg_pos = Z[0].mean()

        amplitude, phase, offset = self.multi_point_centre(
            np.array(z).flatten(), omega_positions
        )
        dy = amplitude * np.sin(phase)
        dx = amplitude * np.cos(phase)

        d = chiRotMatrix.transpose() * np.matrix([[avg_pos], [offset]])

        d_horizontal = d[0] - (self.beam_position[0] / self.zoom.pixels_per_mm)
        d_vertical = d[1] - (self.beam_position[1] / self.zoom.pixels_per_mm)

        # NOTE: We drive alignment x to 0.434 as it corresponds to a
        # focused sample on the MD3
        yield from mv(
            self.sample_x,
            self.sample_x.position + dx,
            self.sample_y,
            self.sample_y.position + dy,
            self.alignment_y,
            self.alignment_y.position + d_vertical[0, 0],
            self.alignment_z,
            self.alignment_z.position - d_horizontal[0, 0],
            self.alignment_x,
            0.434,
        )

    def multi_point_centre(self, z: npt.NDArray, omega_list: list):
        """
        Multipoint centre function

        Parameters
        ----------
        z : npt.NDArray
            A numpy array containing a list of z values obtained during
            three-click centering
        omega_list : list
            A list containing a list of omega values, generally
            [0, 90, 180]

        Returns
        -------
        npt.NDArray
            The optimised parameters: (amplitude, phase, offset)
        """

        optimised_params, _ = optimize.curve_fit(
            self._three_click_centering_func, omega_list, z, p0=[1.0, 0.0, 0.0]
        )

        return optimised_params

    def _three_click_centering_func(
        self, theta: float, amplitude: float, phase: float, offset: float
    ) -> float:
        """
        Sine function used to determine the motor positions at which a sample
        is aligned with the center of the beam.

        Note that the period of the sine function in this case is T=2*pi, therefore
        omega = 2 * pi / T = 1, so the simplified equation we fit is:

        result = amplitude*np.sin(omega*theta + phase) + offset
               = amplitude*np.sin(theta + phase) + offset

        Parameters
        ----------
        theta : float
            Angle in radians
        amplitude : float
            Amplitude of the sine function
        phase : float
            Phase
        offset : float
            Offset

        Returns
        -------
        float
            The value of the sine function at a given amplitude, phase and offset
        """
        return amplitude * np.sin(theta + phase) + offset

    def variance_local_maximum(
        self,
        focus_motor: MD3Motor,
        a: float = 0.0,
        b: float = 1.0,
        tol: float = 0.2,
    ) -> float:
        """
        We use the Golden-section search to find the local maximum of the variance function
        described in the calculate_variance method ( `var( Img * L(x,y) )` ).
        NOTE: We assume that the function is strictly unimodal on [a,b].
        See for example: https://en.wikipedia.org/wiki/Golden-section_search

        Parameters
        ----------
        focus_motor : MD3Motor
            An MD3 motor, can be either a combination of sample_x and
            sample_y, or alignment_x
        a : float
            Minimum value to search for the maximum of var( Img * L(x,y) )
        b : float
            Maximum value to search for the maximum of var( Img * L(x,y) )
        tol : float, optional
            The tolerance, by default 0.2

        Returns
        -------
        Generator[Msg, None, None]
            Moves sample_y to a position where the image is focused
        """
        gr = (np.sqrt(5) + 1) / 2

        c = b - (b - a) / gr
        d = a + (b - a) / gr

        count = 0
        logger.info("Focusing image...")
        while abs(b - a) > tol:
            yield from mv(focus_motor, c)
            val_c = self.calculate_variance()

            yield from mv(focus_motor, d)
            val_d = self.calculate_variance()

            if val_c > val_d:  # val_c > val_d to find the maximum
                b = d
            else:
                a = c

            # We recompute both c and d here to avoid loss of precision which
            # may lead to incorrect results or infinite loop
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            count += 1
            logger.info(f"Iteration: {count}")
        maximum = (b + a) / 2
        logger.info(f"Optimal sample_y value: {maximum}")
        yield from mv(focus_motor, maximum)

    def unblur_image(
        self,
        focus_motor: MD3Motor,
        a: float = 0.0,
        b: float = 1.0,
        tol: float = 0.2,
        number_of_intervals: int = 2,
    ):
        """
        We use the Golden-section search to find the global maximum of the variance function
        described in the calculate_variance method ( `var( Img * L(x,y) )` )
        (see the definition of self.variance_local_maximum).
        In order to find the global maximum, we search for local maximums in N number of
        sub-intervals defined by number_of_intervals.

        Parameters
        ----------
        motor : MD3Motor
            An MD3 motor. We can focus the image with either alignment x, or sample_x and
            sample_y (depending on the value of omega)
        a : float, optional
            Minimum value to search for the maximum of var( Img * L(x,y) )
        b : float, optional
            Maximum value to search for the maximum of var( Img * L(x,y) )
        tol : float, optional
            The tolerance, by default 0.2
        number_of_intervals : int, optional
            Number of sub-intervals used to find the global maximum of a multimodal function

        Yields
        ------
        Generator[Msg, None, None]
            Moves the focus motor to a position where the image is focused
        """

        # Create sub-intervals to find the global maximum
        step = (b - a) / number_of_intervals
        interval_list = []
        for i in range(number_of_intervals):
            interval_list.append((a + step * i, a + step * (i + 1)))

        # Calculate local maximums
        laplacian_list = []
        focus_motor_pos_list = []
        for interval in interval_list:
            yield from self.variance_local_maximum(
                focus_motor, interval[0], interval[1], tol
            )
            laplacian_list.append(self.calculate_variance())
            focus_motor_pos_list.append(focus_motor.position)

        # Find global maximum, and move the focus motor to the best focused position
        argmax = np.argmax(np.array(laplacian_list))
        yield from mv(focus_motor, focus_motor_pos_list[argmax])

    def drive_motors_to_loop_edge(self) -> Generator[Msg, None, None]:
        """
        Drives sample_x and alignment_y to the edge of the loop. The edge of the loop is found
        using the PSI loop finder code

        Yields
        ------
        Generator[Msg, None, None]
            A message that tells bluesky to move the motors to the edge of the loop
        """
        x_coord, y_coord = self.find_loop_edge_coordinates()

        loop_position_sample_x = (
            self.sample_x.position
            - np.sin(np.radians(self.omega.position))
            * (x_coord - self.beam_position[0])
            / self.zoom.pixels_per_mm
        )

        loop_position_sample_y = (
            self.sample_y.position
            - np.cos(np.radians(self.omega.position))
            * (x_coord - self.beam_position[0])
            / self.zoom.pixels_per_mm
        )

        loop_position_alignment_y = (
            self.alignment_y.position
            + (y_coord - self.beam_position[1]) / self.zoom.pixels_per_mm
        )
        yield from mv(
            self.sample_x,
            loop_position_sample_x,
            self.sample_y,
            loop_position_sample_y,
            self.alignment_y,
            loop_position_alignment_y,
        )

    def find_loop_edge_coordinates(self) -> tuple[float, float]:
        """
        We find the edge of the loop using loop finder code developed by PSI.

        Returns
        -------
        tuple[float, float]
            The x and y pixel coordinates of the edge of the loop,
        """
        data = self.get_image_from_md3_camera(np.uint8)

        procImg = loopImageProcessing(data)
        procImg.findContour(
            zoom=self.loop_img_processing_zoom,
            beamline=self.loop_img_processing_beamline,
        )
        extremes = procImg.findExtremes()
        screen_coordinates = extremes["top"]

        x_coord = screen_coordinates[0]
        y_coord = screen_coordinates[1]

        if self.plot:
            omega_pos = round(self.omega.position)
            self.save_image(
                data,
                x_coord,
                y_coord,
                f"{self.sample_id}_loop_centering_{omega_pos}_zoom_{self.zoom.get()}",
            )

        return x_coord, y_coord

    def drive_motors_to_center_of_loop(
        self,
    ) -> Generator[Msg, None, None]:
        """
        Drives the motors to the center of the loop.

        Yields
        ------
        Generator[Msg, None, None]
            A plan that automatically centers a loop
        """

        data = self.get_image_from_md3_camera(np.uint8)

        procImg = loopImageProcessing(data)
        procImg.findContour(
            zoom=self.loop_img_processing_zoom,
            beamline=self.loop_img_processing_beamline,
        )
        procImg.findExtremes()
        rectangle_coordinates = procImg.fitRectangle()

        pos_x_pixels = (
            rectangle_coordinates["top_left"][0]
            + rectangle_coordinates["bottom_right"][0]
        ) / 2
        pos_z_pixels = (
            rectangle_coordinates["top_left"][1]
            + rectangle_coordinates["bottom_right"][1]
        ) / 2

        if self.plot:
            self.plot_raster_grid_and_center_of_loop(
                rectangle_coordinates,
                (pos_x_pixels, pos_z_pixels),
                f"{self.sample_id}_centered_loop",
            )

        loop_position_x = (
            self.sample_x.position
            + (pos_x_pixels - self.beam_position[0]) / self.zoom.pixels_per_mm
        )
        loop_position_z = (
            self.alignment_y.position
            + (pos_z_pixels - self.beam_position[1]) / self.zoom.pixels_per_mm
        )
        yield from mv(self.sample_x, loop_position_x, self.alignment_y, loop_position_z)

    def calculate_variance(self) -> float:
        """
        We calculate the variance of the convolution of the laplacian kernel with an image,
        e.g. var( Img * L(x,y) ), where Img is an image taken from the camera ophyd object,
        and L(x,y) is the Laplacian kernel.

        Returns
        -------
        float
            var( Img * L(x,y) )
        """
        data = self.get_image_from_md3_camera()

        try:
            gray_image = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
        except cv2.error:
            # The MD3 camera already returns black and white images for the zoom levels
            # 5, 6 and 7, so we don't do anything here
            gray_image = data

        return cv2.Laplacian(gray_image, cv2.CV_64F).var()

    def get_image_from_md3_camera(
        self, dtype: npt.DTypeLike = np.uint16, reshape: bool = False
    ) -> npt.NDArray:
        """
        Gets a frame from the camera an reshapes it as
        (height, width, depth).

        Parameters
        ----------
        dtype : npt.DTypeLike, optional
            The data type of the numpy array, by default np.uint16
        reshape : bool, optional
            Reshapes the data to (height, width, depth). The md_camera already returns a
            numpy array of the aforementioned shape, therefore reshape is set to False
            by default

        Returns
        -------
        npt.NDArray
            A frame of shape (height, width, depth)
        """
        try:
            array_data: npt.NDArray = self.md3_camera.array_data.get()
            if reshape:
                data = array_data.reshape(
                    self.md3_camera.height.get(),
                    self.md3_camera.width.get(),
                    self.md3_camera.depth.get(),
                ).astype(dtype)
            else:
                data = array_data.astype(dtype)
        except ConnectionTimeoutError:
            # When the camera is not working, we stream a static image
            # of the test rig
            data = np.load("/mnt/shares/smd_share/blackfly_cam_images/flat.py").astype(
                dtype
            )
            logger.info(
                "WARNING! Camera connection timed out, sending a static image of the test rig"
            )

        return data

    def find_edge_and_flat_angles(self) -> Generator[Msg, None, None]:
        """
        Finds maximum and minimum area of a loop corresponding to the edge and
        flat angles of a loop by calculating. The data is then and normalized and
        fitted to a sine wave assuming that the period T of the sine function is known
        (T=pi by definition)

        Yields
        ------
        Generator[Msg, None, None]
            A bluesky generator
        """
        omega_list = np.arange(0, 360, 45)  # degrees
        area_list = []
        x_axis_error_list = []
        y_axis_error_list = []

        # We zoom in and increase the backlight intensity to improve accuracy
        yield from mv(self.zoom, 4, self.backlight, 2)

        for omega in omega_list:
            yield from mv(self.omega, omega)

            image = self.get_image_from_md3_camera(np.uint8)
            procImg = loopImageProcessing(image)
            procImg.findContour(
                zoom=self.loop_img_processing_zoom,
                beamline=self.loop_img_processing_beamline,
            )
            extremes = procImg.findExtremes()

            if self.plot:
                self.save_image(
                    image,
                    extremes["top"][0],
                    extremes["top"][1],
                    f"{self.sample_id}_area_estimation_{round(self.omega.get())}",
                )
            # NOTE: The area can also be calculated via procImg.contourArea().
            # However, our method `self.quadrilateral_area` seems to be more consistent
            area_list.append(self.quadrilateral_area(extremes))

            error = extremes["top"] - self.beam_position
            x_axis_error_list.append(error[0])
            y_axis_error_list.append(error[1])

        median_x = np.median(x_axis_error_list)
        sigma_x = np.std(x_axis_error_list)
        median_y = np.median(y_axis_error_list)
        sigma_y = np.std(y_axis_error_list)

        if abs(median_x) > 15 or sigma_x > 30 or abs(median_y) > 2 or sigma_y > 2:
            successful_centering = False
            self._plot_histograms(x_axis_error_list, y_axis_error_list)
            logger.info("Optical loop centering has probably failed, aborting workflow")
            return successful_centering
        else:
            successful_centering = True

        # Remove nans from list, and normalize the data (we do not care about amplitude,
        # we only care about phase)
        non_nan_args = np.invert(np.isnan(np.array(area_list)))
        omega_list = omega_list[non_nan_args]
        area_list = np.array(area_list)[non_nan_args]
        area_list = area_list / np.linalg.norm(area_list)

        # Fit the curve
        optimised_params, _ = optimize.curve_fit(
            self._sine_function,
            np.radians(omega_list),
            np.array(area_list),
            p0=[0.2, 0.2, 0],
            maxfev=4000,
        )

        x_new = np.linspace(0, 2 * np.pi, 4096)  # radians
        y_new = self._sine_function(
            x_new, optimised_params[0], optimised_params[1], optimised_params[2]
        )

        argmax = np.argmax(y_new)
        argmin = np.argmin(y_new)

        self.flat_angle = np.degrees(x_new[argmax])
        self.edge_angle = np.degrees(x_new[argmin])

        logger.info(f"Flat angle:  {self.flat_angle}")
        logger.info(f"Edge angle: {self.edge_angle}")

        if self.plot:
            plt.figure()
            plt.plot(x_new, y_new, label="Curve fit")
            plt.plot(np.radians(omega_list), np.array(area_list), label="Data")
            plt.xlabel("Omega [radians]")
            plt.ylabel("Area [pixels^2]")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{self.sample_id}_loop_area_curve_fit")
            plt.close()

            self._plot_histograms(x_axis_error_list, y_axis_error_list)

        return successful_centering

    def _plot_histograms(
        self, x_axis_error_list: list, y_axis_error_list: list
    ) -> None:
        """
        Plots histograms of x_axis_error_list and y_axis_error_list, which
        correspond to the difference between the centered position and beam position,
        i.e. centered_position - beam_position

        Parameters
        ----------
        x_axis_error_list : list
            x axis error list
        y_axis_error_list : list
            y axis error list

        Returns
        -------
        None
        """
        median_x = round(np.median(x_axis_error_list), 1)
        sigma_x = round(np.std(x_axis_error_list), 1)
        median_y = round(np.median(y_axis_error_list), 1)
        sigma_y = round(np.std(y_axis_error_list), 1)
        bins = np.linspace(
            min([min(x_axis_error_list), min(y_axis_error_list)]),
            max([max(x_axis_error_list), max(y_axis_error_list)]),
            6,
        )

        plt.figure()
        plt.hist(
            x_axis_error_list,
            label=f"X axis: $\mu={median_x}$, $\sigma = {sigma_x}$",
            bins=bins,
            histtype="step",
        )
        plt.hist(
            y_axis_error_list,
            label=f"Y axis: $\mu={median_y}$, $\sigma = {sigma_y}$",
            bins=bins,
            histtype="step",
            linestyle="--",
        )
        plt.xlabel("(Centered position - Beam position) [pixels]")
        plt.ylabel("Counts")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.sample_id}_optical_centering_accuracy")
        plt.close()

    def _sine_function(
        self, theta: float, amplitude: float, phase: float, offset: float
    ) -> float:
        """
        Sine function used to find the angles at which the area of a loop
        is maximum and minimum:

        area = amplitude*np.sin(omega*theta + phase) + offset

        Note that the period of the sine function is in this case T=pi, therefore
        omega = 2 * pi / T = 2, so the simplified equation we fit is:

        area = amplitude*np.sin(2*theta + phase) + offset

        Parameters
        ----------
        theta : float
            Angle in radians
        amplitude : float
            Amplitude of the sine function
        phase : float
            Phase
        offset : float
            Offset

        Returns
        -------
        float
            The area of the loop at a given angle
        """

        return amplitude * np.sin(2 * theta + phase) + offset

    def move_loop_to_md3_field_of_view(self) -> Generator[Msg, None, None]:
        """
        We use the top camera to move the loop to the md3 camera field of view.
        x_pixel_target and y_pixel_target are the pixel coordinates that correspond
        to the position where the loop is seen fully by the md3 camera. These
        values are calculated experimentally and must be callibrated every time the top
        camera is moved.

        Yields
        ------
        Generator[Msg, None, None]
            A bluesky generator
        """

        yield from mv(self.omega, 0, self.zoom, 1, self.backlight, 2)

        # Check if there is a pin or not
        img = self.top_camera.array_data.get()

        std = np.std(img - top_camera_background_img_array)
        # The number 2 is determined experimentally
        if std < 2:
            logger.info(
                "No pin found during the pre-centering step. "
                "Optical and x-ray centering will not continue"
            )
            loop_found = False
            return loop_found
        else:
            loop_found = True

        img = img.reshape(
            self.top_camera.height.get(), self.top_camera.width.get()
        ).astype(np.uint8)
        img = img[
            self.top_camera_roi_y[0] : self.top_camera_roi_y[1],
            self.top_camera_roi_x[0] : self.top_camera_roi_x[1],
        ]

        procImg = loopImageProcessing(img)
        procImg.findContour(
            zoom="top_camera",
            beamline="MX3",
        )
        screen_coordinates = procImg.findExtremes()["top"]

        x_coord = screen_coordinates[0]
        y_coord = screen_coordinates[1]
        if self.plot:
            self.save_image(
                img,
                x_coord,
                y_coord,
                f"{self.sample_id}_top_camera",
                grayscale_img=True,
            )

        delta_mm_x = (self.x_pixel_target - x_coord) / self.top_camera.pixels_per_mm_x
        delta_mm_y = (self.y_pixel_target - y_coord) / self.top_camera.pixels_per_mm_y
        yield from mv(
            self.alignment_y,
            self.alignment_y.position - delta_mm_y,
            self.sample_y,
            self.sample_y.position - delta_mm_x,
        )

        return loop_found

    def save_image(
        self,
        data: npt.NDArray,
        x_coord: float,
        y_coord: float,
        filename: str,
        grayscale_img: bool = False,
    ) -> None:
        """
        Saves an image from a numpy array taken from the camera ophyd object,
        and draws a red cross at the screen_coordinates.

        Parameters
        ----------
        data : npt.NDArray
            A numpy array containing an image from the camera
        x_coord : float
            X coordinate
        y_coord : float
            Y coordinate
        grayscale_img : bool
            If the image is in grayscale, set this value to True, by default False
        filename : str
            The filename

        Returns
        -------
        None
        """
        plt.figure()
        if grayscale_img:
            plt.imshow(data, cmap="gray", vmin=0, vmax=255)
        else:
            plt.imshow(data)
        plt.scatter(
            x_coord,
            y_coord,
            s=200,
            c="r",
            marker="+",
            label=f"Omega={self.omega.position}",
        )
        plt.legend()
        plt.savefig(filename)
        plt.close()

    def plot_raster_grid_and_center_of_loop(
        self,
        rectangle_coordinates: dict,
        loop_center_coordinates: tuple[int, int],
        filename: str,
    ) -> None:
        """
        Plots the limits of the raster grid on top of the image taken from the
        camera as well of the center of the raster grid.

        Parameters
        ----------
        rectangle_coordinates : dict
            The coordinates of the rectangle surrounding the loop
        loop_center_coordinates : tuple[int, int]
            Center of the loop coordinates

        Returns
        -------
        None
        """
        plt.figure()
        data = self.get_image_from_md3_camera()
        plt.imshow(data)

        # Plot Rectangle coordinates
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

        # Plot grid:
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

        # Plot center of the loop
        plt.scatter(
            loop_center_coordinates[0],
            loop_center_coordinates[1],
            s=200,
            c="r",
            marker="+",
        )
        plt.savefig(filename)
        plt.close()

    def magnitude(self, vector: npt.NDArray) -> npt.NDArray:
        """Calculates the magnitude of a vector

        Parameters
        ----------
        vector : npt.NDArray
            A numpy array vector

        Returns
        -------
        npt.NDArray
            The magnitude of a vector
        """
        return np.sqrt(np.dot(vector, vector))

    def quadrilateral_area(self, extremes: dict) -> float:
        """
        Area of a quadrilateral. For details on how to calculate the area of a
        quadrilateral see e.g.
        https://byjus.com/maths/area-of-quadrilateral/

        Parameters
        ----------
        extremes : dict
            A dictionary containing four extremes of a loop. This dictionary is assumed to
            be the extremes returned by the loop finder code developed by PSI
            (see the findExtremes method of the psi code)

        Returns
        -------
        float
            The area of a quadrilateral
        """
        a = np.sqrt(
            (extremes["bottom"][0] - extremes["right"][0]) ** 2
            + (extremes["bottom"][1] - extremes["right"][1]) ** 2
        )
        b = np.sqrt(
            (extremes["bottom"][0] - extremes["left"][0]) ** 2
            + (extremes["bottom"][1] - extremes["left"][1]) ** 2
        )
        c = np.sqrt(
            (extremes["top"][0] - extremes["left"][0]) ** 2
            + (extremes["top"][1] - extremes["left"][1]) ** 2
        )
        d = np.sqrt(
            (extremes["top"][0] - extremes["right"][0]) ** 2
            + (extremes["top"][1] - extremes["right"][1]) ** 2
        )

        s = (a + b + c + d) / 2

        a_vector = extremes["right"] - extremes["bottom"]
        b_vector = extremes["left"] - extremes["bottom"]
        c_vector = extremes["left"] - extremes["top"]
        d_vector = extremes["right"] - extremes["top"]

        theta_1 = np.arccos(
            np.dot(a_vector, b_vector)
            / (self.magnitude(a_vector) * self.magnitude(b_vector))
        )
        theta_2 = np.arccos(
            np.dot(c_vector, d_vector)
            / (self.magnitude(c_vector) * self.magnitude(d_vector))
        )

        theta = theta_1 + theta_2

        area = np.sqrt(
            (s - a) * (s - b) * (s - c) * (s - d)
            - a * b * c * d * (np.cos(theta / 2)) ** 2
        )

        return area

    def prepare_raster_grid(
        self, omega: float, filename: str = "step_3_prep_raster"
    ) -> tuple[RasterGridMotorCoordinates, dict]:
        """
        Prepares a raster grid. The limits of the grid are obtained using
        the PSI loop centering code

        Parameters
        ----------
        omega : float
            Angle at which the grid scan is done
        filename: str
            Name of the file used to plot save the results if self.plot = True,
            by default step_3_prep_raster

        Returns
        -------
        motor_coordinates: RasterGridMotorCoordinates
            A pydantic model containing the initial and final motor positions of the grid.
        rectangle_coordinates: dict
            Rectangle coordinates in pixels
        """
        # the loopImageProcessing code only works with np.uint8 data types
        data = self.get_image_from_md3_camera(np.uint8)

        procImg = loopImageProcessing(data)
        procImg.findContour(
            zoom=self.loop_img_processing_zoom,
            beamline=self.loop_img_processing_beamline,
        )
        procImg.findExtremes()
        rectangle_coordinates = procImg.fitRectangle()

        if self.plot:
            self.plot_raster_grid(
                rectangle_coordinates,
                filename,
            )

        # Width and height of the grid in (mm)
        width = (
            abs(
                rectangle_coordinates["top_left"][0]
                - rectangle_coordinates["bottom_right"][0]
            )
            / self.zoom.pixels_per_mm
        )
        height = (
            abs(
                rectangle_coordinates["top_left"][1]
                - rectangle_coordinates["bottom_right"][1]
            )
            / self.zoom.pixels_per_mm
        )

        # Y pixel coordinates
        initial_pos_y_pixels = abs(
            rectangle_coordinates["top_left"][1] - self.beam_position[1]
        )
        final_pos_y_pixels = abs(
            rectangle_coordinates["bottom_right"][1] - self.beam_position[1]
        )

        # Alignment y target positions (mm)
        initial_pos_alignment_y = (
            self.alignment_y.position - initial_pos_y_pixels / self.zoom.pixels_per_mm
        )
        final_pos_alignment_y = (
            self.alignment_y.position + final_pos_y_pixels / self.zoom.pixels_per_mm
        )

        # X pixel coordinates
        initial_pos_x_pixels = abs(
            rectangle_coordinates["top_left"][0] - self.beam_position[0]
        )
        final_pos_x_pixels = abs(
            rectangle_coordinates["bottom_right"][0] - self.beam_position[0]
        )

        # Sample x target positions (mm)
        initial_pos_sample_x = self.sample_x.position - np.sin(
            np.radians(self.omega.position)
        ) * (initial_pos_x_pixels / self.zoom.pixels_per_mm)
        final_pos_sample_x = self.sample_x.position + np.sin(
            np.radians(self.omega.position)
        ) * (+final_pos_x_pixels / self.zoom.pixels_per_mm)

        # Sample y target positions (mm)
        initial_pos_sample_y = self.sample_y.position - np.cos(
            np.radians(self.omega.position)
        ) * (initial_pos_x_pixels / self.zoom.pixels_per_mm)
        final_pos_sample_y = self.sample_y.position + np.cos(
            np.radians(self.omega.position)
        ) * (final_pos_x_pixels / self.zoom.pixels_per_mm)

        # Center of the grid (mm) (y-axis only)
        center_x_of_grid_pixels = (
            rectangle_coordinates["top_left"][0]
            + rectangle_coordinates["bottom_right"][0]
        ) / 2
        center_pos_sample_x = self.sample_x.position + np.sin(
            np.radians(self.omega.position)
        ) * (
            (center_x_of_grid_pixels - self.beam_position[0]) / self.zoom.pixels_per_mm
        )
        center_pos_sample_y = self.sample_y.position + np.cos(
            np.radians(self.omega.position)
        ) * (
            (center_x_of_grid_pixels - self.beam_position[0]) / self.zoom.pixels_per_mm
        )

        # NOTE: The width and height are measured in mm and the beam_size in micrometers,
        # hence the conversion below
        number_of_columns = int(width / (self.beam_size[0] / 1000))
        number_of_rows = int(height / (self.beam_size[1] / 1000))

        motor_coordinates = RasterGridMotorCoordinates(
            initial_pos_sample_x=initial_pos_sample_x,
            final_pos_sample_x=final_pos_sample_x,
            initial_pos_sample_y=initial_pos_sample_y,
            final_pos_sample_y=final_pos_sample_y,
            initial_pos_alignment_y=initial_pos_alignment_y,
            final_pos_alignment_y=final_pos_alignment_y,
            width=width,
            height=height,
            center_pos_sample_x=center_pos_sample_x,
            center_pos_sample_y=center_pos_sample_y,
            number_of_columns=number_of_columns,
            number_of_rows=number_of_rows,
            omega=omega,
        )
        logger.info(f"Raster grid coordinates [mm]: {motor_coordinates}")

        return motor_coordinates, rectangle_coordinates

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
        initial_pos_pixels: list[int, int]
            The x and z coordinates of the initial position of the grid
        final_pos_pixels: list[int, int]
            The x and z coordinates of the final position of the grid
        filename: str
            The name of the PNG file

        Returns
        -------
        None
        """
        plt.figure()
        data = self.get_image_from_md3_camera()
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
        plt.title(f"Omega = {round(self.omega.position, 2)} [degrees]")
        plt.legend()

        plt.savefig(filename)
        plt.close()


path_to_config_file = path.join(
    path.dirname(__file__), "configuration/optical_and_xray_centering.yml"
)

with open(path_to_config_file, "r") as plan_config:
    plan_args: dict = yaml.safe_load(plan_config)


def optical_centering(
    sample_id: str,
    md3_camera: MDRedisCam,
    top_camera: BlackFlyCam,
    sample_x: Union[CosylabMotor, MD3Motor],
    sample_y: Union[CosylabMotor, MD3Motor],
    alignment_x: Union[CosylabMotor, MD3Motor],
    alignment_y: Union[CosylabMotor, MD3Motor],
    alignment_z: Union[CosylabMotor, MD3Motor],
    omega: Union[CosylabMotor, MD3Motor],
    zoom: MD3Zoom,
    phase: MD3Phase,
    backlight: MD3BackLight,
    beam_position: tuple[int, int],
    beam_size: tuple[float, float],
    top_camera_background_img_array: npt.NDArray = None,
):
    """
    Parameters
    ----------
    sample_id : str
     Sample id
    md3_camera : MDRedisCam
        MD3 Camera
    top_camera: BlackFlyCam
        Top Camera
    sample_x : Union[CosylabMotor, MD3Motor]
        Sample x
    sample_y : Union[CosylabMotor, MD3Motor]
        Sample y
    alignment_x : Union[CosylabMotor, MD3Motor]
        Alignment x
    alignment_y : Union[CosylabMotor, MD3Motor]
        Alignment y
    alignment_z : Union[CosylabMotor, MD3Motor]
        Alignment y
    omega : Union[CosylabMotor, MD3Motor]
        Omega
    zoom : MD3Zoom
        Zoom
    phase : MD3Phase
        MD3 phase ophyd-signal
    backlight : MD3Backlight
        Backlight
    beam_position : tuple[int, int]
        Position of the beam
    beam_size : tuple[float, float]
        Beam size
    top_camera_background_img_array : npt.NDArray, optional
        Top camera background image array used to determine if there is a pin.
        If top_camera_background_img_array is None, we use the default background image from
        the mx3-beamline-library

    Returns
    -------
    None
    """

    loop_img_processing_beamline: str = plan_args["loop_image_processing"]["beamline"]
    loop_img_processing_zoom: str = plan_args["loop_image_processing"]["zoom"]
    auto_focus: bool = plan_args["autofocus_image"]["autofocus"]
    min_focus: float = plan_args["autofocus_image"]["min"]
    max_focus: float = plan_args["autofocus_image"]["max"]
    tol: float = plan_args["autofocus_image"]["tol"]
    plot: bool = plan_args["plot_results"]
    number_of_intervals: float = plan_args["autofocus_image"]["number_of_intervals"]
    number_of_omega_steps: float = plan_args["loop_area_estimation"][
        "number_of_omega_steps"
    ]
    x_pixel_target: int = plan_args["top_camera"]["x_pixel_target"]
    y_pixel_target: int = plan_args["top_camera"]["y_pixel_target"]
    top_camera_roi_x: tuple[int, int] = tuple(plan_args["top_camera"]["roi_x"])
    top_camera_roi_y: tuple[int, int] = tuple(plan_args["top_camera"]["roi_y"])

    _optical_centering = OpticalCentering(
        sample_id=sample_id,
        md3_camera=md3_camera,
        top_camera=top_camera,
        sample_x=sample_x,
        sample_y=sample_y,
        alignment_x=alignment_x,
        alignment_y=alignment_y,
        alignment_z=alignment_z,
        omega=omega,
        zoom=zoom,
        phase=phase,
        backlight=backlight,
        beam_position=beam_position,
        beam_size=beam_size,
        auto_focus=auto_focus,
        min_focus=min_focus,
        max_focus=max_focus,
        tol=tol,
        number_of_intervals=number_of_intervals,
        plot=plot,
        loop_img_processing_beamline=loop_img_processing_beamline,
        loop_img_processing_zoom=loop_img_processing_zoom,
        number_of_omega_steps=number_of_omega_steps,
        x_pixel_target=x_pixel_target,
        y_pixel_target=y_pixel_target,
        top_camera_background_img_array=top_camera_background_img_array,
        top_camera_roi_x=top_camera_roi_x,
        top_camera_roi_y=top_camera_roi_y,
    )

    yield from monitor_during_wrapper(
        run_wrapper(_optical_centering.center_loop(), md={"sample_id": sample_id}),
        signals=(
            sample_x,
            sample_y,
            alignment_x,
            alignment_y,
            alignment_z,
            omega,
            phase,
            backlight,
            _optical_centering.grid_scan_coordinates_edge,
            _optical_centering.grid_scan_coordinates_flat,
        ),
    )
