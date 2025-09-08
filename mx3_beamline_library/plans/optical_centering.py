import pickle
from io import BytesIO
from os import getcwd, makedirs, path
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from bluesky.plan_stubs import mv
from bluesky.preprocessors import monitor_during_wrapper, run_wrapper
from bluesky.tracing import trace_plan, tracer
from bluesky.utils import Msg
from matplotlib import rc
from ophyd import Signal
from PIL import Image
from scipy import optimize
from scipy.stats import kstest

from ..config import BL_ACTIVE, redis_connection
from ..constants import top_camera_background_img_array
from ..devices.detectors import blackfly_camera, md3_camera
from ..devices.motors import md3
from ..logger import setup_logger
from ..schemas.optical_centering import (
    CenteredLoopMotorCoordinates,
    OpticalCenteringExtraConfig,
    OpticalCenteringResults,
    TopCameraConfig,
)
from ..schemas.xray_centering import RasterGridCoordinates
from ..science.optical_and_loop_centering.loop_edge_detection import LoopEdgeDetection
from .image_analysis import (
    get_image_from_md3_camera,
    get_image_from_top_camera,
    unblur_image_fast,
)
from .plan_stubs import md3_move

logger = setup_logger()

rc("xtick", labelsize=15)
rc("ytick", labelsize=15)
# Set usetex=True to get nice LATEX plots. Note that
# using LATEX significantly reduces speed!
rc("text", usetex=False)


class OpticalCentering:
    """
    This class runs a bluesky plan that optically aligns the loop with the
    center of the beam. Before analysing an image, we unblur the image at the start
    of the plan to make sure the results are consistent. Finally we find angles at which
    the area of a loop is maximum and minimum (flat and edge) and we calculate the grid
    coordinates for the flat and edge angles.
    """

    def __init__(
        self,
        sample_id: int,
        beam_position: tuple[int, int],
        grid_step: tuple[float, float] | None = None,
        calibrated_alignment_z: float = 0.634,
        plot: bool = False,
        top_camera_background_img_array: npt.NDArray = None,
        output_directory: str | None = None,
        use_top_camera_camera: bool = True,
        manual_mode: bool = False,
        extra_config: OpticalCenteringExtraConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------
        sample_id : int
            Sample id
        beam_position : tuple[int, int]
            Position of the beam
        grid_step : tuple[float, float] | None
            The step of the grid (x,y) in micrometers. Can also be None
            only if manual_mode=True
        calibrated_alignment_z : float, optional.
            The alignment_z position which aligns a sample with the center of rotation
            at the beam position. This value is calculated experimentally, by default
            0.662
        plot : bool, optional
            If true, we take snapshots of the loop at different stages
            of the plan, by default False
        top_camera_background_img_array : npt.NDArray, optional
            Top camera background image array used to determine if there is a pin.
            If top_camera_background_img_array is None, we use the default background image from
            the mx3-beamline-library
        output_directory : str | None, optional
            The directory where all diagnostic plots are saved if self.plot=True.
            If output_directory=None, we use the current working directory,
            by default None
        use_top_camera_camera : bool, optional
            Determines if we use the top camera (a.k.a. zoom level 0) for loop centering.
            This flag should only be set to False for development purposes, or when
            the top camera is not working. By default True
        manual_mode : bool, optional
            Determine if optical centering is run manual mode (e.g. from mxcube).
            In this case, we only align the loop with the center of the beam,
            but we do not infer the coordinates used for rastering. The results are not
            saved to redis, by default False.
        extra_config : OpticalCenteringExtraConfig | None, optional
            The optical centering extra configuration. This contains configuration that
            does not change often. If extra_config is None, the default value is set to
            OpticalCenteringExtraConfig()

        Returns
        -------
        None
        """
        self.sample_id = sample_id
        self.md3_camera = md3_camera
        self.top_camera = blackfly_camera
        self.beam_position = beam_position
        self.grid_step = grid_step
        self.plot = plot
        self.calibrated_alignment_z = calibrated_alignment_z

        self._check_top_camera_config()
        self.centered_loop_coordinates = None

        self._set_optical_centering_config_parameters(extra_config)

        self.grid_scan_coordinates_flat = Signal(
            name="grid_scan_coordinates_flat", kind="normal"
        )
        self.grid_scan_coordinates_edge = Signal(
            name="grid_scan_coordinates_edge", kind="normal"
        )
        if top_camera_background_img_array is None:
            self.top_camera_background_img_array = top_camera_background_img_array

        if output_directory is None:
            self.output_directory = getcwd()
        else:
            self.output_directory = output_directory

        self.sample_path = path.join(self.output_directory, str(self.sample_id))
        if self.plot:
            try:
                makedirs(self.sample_path)
            except FileExistsError:
                pass
        self.use_top_camera_camera = use_top_camera_camera
        self.manual_mode = manual_mode

        if not self.manual_mode:
            if grid_step is None:
                raise ValueError("grid_step can only be None if manual_mode=True")

        self.final_zoom_level = 4

    def _check_top_camera_config(self) -> None:
        """
        Checks that the top camera is configured correctly. If it is not,
        PV values are updated following the TopCameraConfig model

        Returns
        -------
        None
        """
        config = TopCameraConfig()
        if BL_ACTIVE == "true":
            if (
                self.top_camera.cc1_enable_callbacks.get()
                != config.cc1.enable_callbacks
            ):
                self.top_camera.cc1_enable_callbacks.set(config.cc1.enable_callbacks)

            if self.top_camera.nd_array_port.get() != config.image.nd_array_port:
                self.top_camera.nd_array_port.set(config.image.nd_array_port)

            if self.top_camera.enable_callbacks.get() != config.cam.enable_callbacks:
                self.top_camera.enable_callbacks.set(config.cam.enable_callbacks)

            if self.top_camera.array_callbacks.get() != config.cam.array_callbacks:
                self.top_camera.array_callbacks.set(config.cam.array_callbacks)

            if self.top_camera.frame_rate_enable.get() != config.cam.frame_rate_enable:
                self.top_camera.frame_rate_enable.set(config.cam.frame_rate_enable)

            if self.top_camera.gain_auto.get() != config.cam.gain_auto:
                self.top_camera.gain_auto.set(config.cam.gain_auto)

            if self.top_camera.exposure_auto.get() != config.cam.exposure_auto:
                self.top_camera.exposure_auto.set(config.cam.exposure_auto)

            if self.top_camera.pixel_format.get() != config.cam.pixel_format:
                self.top_camera.pixel_format.set(config.cam.pixel_format)

            if round(self.top_camera.frame_rate.get(), 1) != config.cam.frame_rate:
                self.top_camera.frame_rate.set(config.cam.frame_rate)

            if round(self.top_camera.gain.get(), 1) != config.cam.gain:
                self.top_camera.gain.set(config.cam.gain)

            if (
                round(self.top_camera.exposure_time.get(), 3)
                != config.cam.exposure_time
            ):
                self.top_camera.exposure_time.set(config.cam.exposure_time)

            if (
                round(self.top_camera.acquire_period.get(), 2)
                != config.cam.acquire_period
            ):
                self.top_camera.acquire_period.set(config.cam.acquire_period)

    def _set_optical_centering_config_parameters(
        self, optical_centering_config: OpticalCenteringExtraConfig | None
    ) -> None:
        """
            Sets the extra configuration values used during optical centering

            Parameters
            ----------
            optical_centering_config : OpticalCenteringExtraConfig | None, optional
            The optical centering extra configuration. This contains configuration that
            does not change often. If extra_config is None, the default value is set to
            OpticalCenteringExtraConfig()

        Returns
            -------
            None
        """
        if optical_centering_config is None:
            optical_centering_config = OpticalCenteringExtraConfig()

        self.md3_cam_block_size = (
            optical_centering_config.md3_camera.loop_image_processing.block_size
        )
        self.md3_cam_adaptive_constant = (
            optical_centering_config.md3_camera.loop_image_processing.adaptive_constant
        )
        self.top_cam_block_size = (
            optical_centering_config.top_camera.loop_image_processing.block_size
        )
        self.top_cam_adaptive_constant = (
            optical_centering_config.top_camera.loop_image_processing.adaptive_constant
        )
        self.alignment_x_default_pos = (
            optical_centering_config.motor_default_positions.alignment_x
        )
        self.top_camera_roi_x = optical_centering_config.top_camera.roi_x
        self.top_camera_roi_y = optical_centering_config.top_camera.roi_y
        self.auto_focus = optical_centering_config.autofocus_image.autofocus
        self.min_focus = optical_centering_config.autofocus_image.min
        self.max_focus = optical_centering_config.autofocus_image.max

        top_cam_target_coords = redis_connection.hgetall("top_camera_target_coords")
        self.x_pixel_target = float(top_cam_target_coords[b"x_pixel_target"])
        self.y_pixel_target = float(top_cam_target_coords[b"y_pixel_target"])
        self.percentage_error = (
            optical_centering_config.optical_centering_percentage_error
        )

        top_cam_pixels_per_mm = redis_connection.hgetall("top_camera_pixels_per_mm")
        self.top_cam_pixels_per_mm_x = float(top_cam_pixels_per_mm[b"pixels_per_mm_x"])
        self.top_cam_pixels_per_mm_y = float(top_cam_pixels_per_mm[b"pixels_per_mm_y"])
        self.grid_height_scale_factor = (
            optical_centering_config.grid_height_scale_factor
        )

    @trace_plan(tracer, "center_loop")
    def center_loop(self) -> Generator[Msg, None, None]:
        """
        Opens and closes the run while keeping track of the signals
        used in the loop centering plans

        Yields
        ------
        Generator[Msg, None, None]
            The loop centering plan generator
        """
        yield from monitor_during_wrapper(
            run_wrapper(self._center_loop(), md={"sample_id": self.sample_id}),
            signals=(
                md3.sample_x,
                md3.sample_y,
                md3.alignment_x,
                md3.alignment_y,
                md3.alignment_z,
                md3.omega,
                md3.phase,
                md3.backlight,
                self.grid_scan_coordinates_edge,
                self.grid_scan_coordinates_flat,
            ),
        )

    def _center_loop(self) -> Generator[Msg, None, None]:
        """
        This plan is the main optical loop centering plan. Before analysing an image.
        we unblur the image at to make sure the results are consistent. After finding the
        centered (motor coordinates) of the loop, we find the edge and flat angles of the
        loop. Finally, the results are saved to redis following the convention:
            f"optical_centering_results:{self.sample_id}"


        Yields
        ------
        Generator[Msg, None, None]
            A plan that automatically centers a loop
        """
        # Set phase to `Centring`
        current_phase = md3.phase.get()
        if current_phase != "Centring":
            yield from mv(md3.phase, "Centring")

        yield from mv(md3.alignment_z, self.calibrated_alignment_z)

        if self.use_top_camera_camera:
            loop_found = yield from self.move_loop_to_md3_field_of_view()
        else:
            loop_found = True

        if not loop_found:
            optical_centering_results = OpticalCenteringResults(
                optical_centering_successful=False
            )
            redis_connection.set(
                f"optical_centering_results:{self.sample_id}",
                pickle.dumps(optical_centering_results.model_dump()),
            )
            raise ValueError("No loop found by the zoom level-0 camera")

        # We center the loop at two different zooms
        yield from mv(md3.zoom, 1)
        yield from self.multi_point_centering_plan()

        successful_centering = yield from self.find_edge_and_flat_angles()
        self.centered_loop_coordinates = CenteredLoopMotorCoordinates(
            alignment_x=md3.alignment_x.position,
            alignment_y=md3.alignment_y.position,
            alignment_z=md3.alignment_z.position,
            sample_x=md3.sample_x.position,
            sample_y=md3.sample_y.position,
        )

        md3.save_centring_position()

        if not self.manual_mode:
            if not successful_centering:
                optical_centering_results = OpticalCenteringResults(
                    optical_centering_successful=False
                )
                redis_connection.set(
                    f"optical_centering_results:{self.sample_id}",
                    pickle.dumps(optical_centering_results.model_dump()),
                )
                raise ValueError("Optical centering was not successful")

            # Prepare grid for the edge surface
            yield from mv(md3.zoom, self.final_zoom_level)
            yield from mv(md3.omega, self.edge_angle)
            filename_edge = path.join(
                self.sample_path, f"{self.sample_id}_raster_grid_edge"
            )
            grid_edge = self.prepare_raster_grid(self.edge_angle, filename_edge)
            # Add metadata for bluesky documents
            yield from mv(self.grid_scan_coordinates_edge, grid_edge.model_dump())

            # Prepare grid for the flat surface
            yield from mv(md3.zoom, self.final_zoom_level)
            yield from mv(md3.omega, self.flat_angle)
            filename_flat = path.join(
                self.sample_path, f"{self.sample_id}_raster_grid_flat"
            )
            grid_flat = self.prepare_raster_grid(self.flat_angle, filename_flat)
            # Add metadata for bluesky documents
            yield from mv(self.grid_scan_coordinates_flat, grid_flat.model_dump())

            optical_centering_results = OpticalCenteringResults(
                optical_centering_successful=True,
                centered_loop_coordinates=self.centered_loop_coordinates,
                edge_angle=self.edge_angle,
                flat_angle=self.flat_angle,
                edge_grid_motor_coordinates=grid_edge,
                flat_grid_motor_coordinates=grid_flat,
            )

            # Save results to redis
            redis_connection.set(
                f"optical_centering_results:{self.sample_id}",
                pickle.dumps(optical_centering_results.model_dump()),
            )
            logger.info("Optical centering successful!")

    def multi_point_centering_plan(self) -> Generator[Msg, None, None]:
        """
        Runs the multi-point centering procedure to align the loop with the center of
        the beam.

        Yields
        ------
        Generator[Msg, None, None]
            A bluesky message
        """
        yield from mv(md3.zoom, 1)
        start_omega = md3.omega.position
        omega_array = (
            np.array([start_omega, start_omega + 90, start_omega + 2 * 90]) % 360
        )
        focused_position_list = []
        for i, omega in enumerate(omega_array):
            if omega != start_omega:
                yield from mv(md3.omega, omega)
            if self.auto_focus:
                if i % 2:
                    focused_alignment_x = yield from unblur_image_fast(
                        md3.alignment_x,
                        start_position=self.min_focus,
                        final_position=self.max_focus,
                    )
                    focused_position_list.append(focused_alignment_x)
                else:
                    focused_alignment_x = yield from unblur_image_fast(
                        md3.alignment_x,
                        start_position=self.max_focus,
                        final_position=self.min_focus,
                    )
                    focused_position_list.append(focused_alignment_x)

        # Start from the last positions (this optimises the plan)
        omega_array = np.flip(omega_array)
        focused_position_list.reverse()

        x_coords, y_coords = [], []
        for omega, alignment_x in zip(omega_array, focused_position_list):
            yield from md3_move(md3.omega, omega, md3.alignment_x, alignment_x)

            x, y = self.find_loop_edge_coordinates()
            x_coords.append(x / md3.zoom.pixels_per_mm)
            y_coords.append(y / md3.zoom.pixels_per_mm)

        yield from self.three_click_centering(
            x_coords, y_coords, np.radians(omega_array)
        )

    def two_click_centering(
        self,
        x_coords: list,
        y_coords: list,
        omega_positions: list,
    ) -> Generator[Msg, None, None]:
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
        average_y_position = np.min(y_coords)

        amplitude, phase = self.multi_point_centre(
            x_coords, omega_positions, two_clicks=True
        )
        delta_sample_y = amplitude * np.sin(phase)
        delta_sample_x = amplitude * np.cos(phase)

        delta_alignment_y = average_y_position - (
            self.beam_position[1] / md3.zoom.pixels_per_mm
        )

        yield from md3_move(
            md3.sample_x,
            md3.sample_x.position + delta_sample_x,
            md3.sample_y,
            md3.sample_y.position + delta_sample_y,
            md3.alignment_y,
            md3.alignment_y.position + delta_alignment_y,
            md3.alignment_z,
            self.calibrated_alignment_z,
            md3.alignment_x,
            self.alignment_x_default_pos,
        )

    def three_click_centering(
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
        average_y_position = np.min(y_coords)

        amplitude, phase, offset = self.multi_point_centre(
            x_coords, omega_positions, two_clicks=False
        )
        delta_sample_y = amplitude * np.sin(phase)
        delta_sample_x = amplitude * np.cos(phase)

        delta_alignment_y = average_y_position - (
            self.beam_position[1] / md3.zoom.pixels_per_mm
        )
        delta_alignment_z = offset - (self.beam_position[0] / md3.zoom.pixels_per_mm)

        yield from md3_move(
            md3.sample_x,
            md3.sample_x.position + delta_sample_x,
            md3.sample_y,
            md3.sample_y.position + delta_sample_y,
            md3.alignment_y,
            md3.alignment_y.position + delta_alignment_y,
            md3.alignment_z,
            md3.alignment_z.position - delta_alignment_z,
            md3.alignment_x,
            self.alignment_x_default_pos,
        )

    def multi_point_centre(
        self, x_coords: list, omega_list: list, two_clicks: bool
    ) -> npt.NDArray:
        """
        Multipoint centre function

        Parameters
        ----------
        x_coords : list
            A list of x-coordinates values obtained during
            three-click centering in mm
        omega_list : list
            A list containing a list of omega values in radians
        two_clicks : bool
            If two_clicks= True, we only fit phase and amplitude,
            otherwise we fit phase, amplitude and offset, in which
            case a minimum of three clicks is needed
        Returns
        -------
        npt.NDArray
            The optimised parameters: (amplitude, phase)
        """
        if two_clicks:
            optimised_params, _ = optimize.curve_fit(
                self.two_click_centering_function, omega_list, x_coords, p0=[1.0, 0.0]
            )
        else:
            optimised_params, _ = optimize.curve_fit(
                self.three_click_centering_function,
                omega_list,
                x_coords,
                p0=[1.0, 0.0, 0.0],
            )

        return optimised_params

    def two_click_centering_function(
        self, theta: float, amplitude: float, phase: float
    ) -> float:
        """
        Sine function used to determine the motor positions at which a sample
        is aligned with the center of the beam.

        Note that the period of the sine function in this case is T=2*pi, therefore
        omega = 2 * pi / T = 1. Additionally the offset is a constant calculated
        experimentally, so effectively we fit a function with two unknowns:
        phase and theta:

        result = amplitude*np.sin(omega*theta + phase) + offset
               = amplitude*np.sin(theta + phase) + offset

        Parameters
        ----------
        theta : float
            Angle in radians
        amplitude : float
            Amplitude of the sine function in mm
        phase : float
            Phase in radians

        Returns
        -------
        float
            The value of the sine function at a given angle, amplitude and phase
        """
        offset = self.beam_position[0] / md3.zoom.pixels_per_mm
        return amplitude * np.sin(theta + phase) + offset

    def three_click_centering_function(
        self, theta: float, amplitude: float, phase: float, offset: float
    ) -> float:
        """
        Sine function used to determine the motor positions at which a sample
        is aligned with the center of the beam.

        Note that the period of the sine function in this case is T=2*pi, therefore
        omega = 2 * pi / T = 1. In this case we additionally estimate the offset

        Parameters
        ----------
        theta : float
            Angle in radians
        amplitude : float
            Amplitude of the sine function in mm
        phase : float
            Phase in radians
        offset : float
            Offset

        Returns
        -------
        float
            The value of the sine function at a given angle, amplitude and phase
        """
        return amplitude * np.sin(theta + phase) + offset

    def find_loop_edge_coordinates(self) -> tuple[float, float]:
        """
        We find the edge of the loop using the LoopEdgeDetection class.

        Returns
        -------
        tuple[float, float]
            The x and y pixel coordinates of the edge of the loop
        """
        x_coord_list = []
        y_coord_list = []
        for _ in range(5):
            data = get_image_from_md3_camera(np.uint8)
            edge_detection = LoopEdgeDetection(
                data,
                block_size=self.top_cam_block_size,
                adaptive_constant=self.top_cam_adaptive_constant,
            )
            tip = edge_detection.find_tip()
            x_coord_list.append(tip[0])
            y_coord_list.append(tip[1])

        x_coord = x_coord_list[np.argmin(x_coord_list)]
        y_coord = np.min(y_coord_list)  # min or max?

        if self.plot:
            omega_pos = round(md3.omega.position)
            filename = path.join(
                self.sample_path,
                f"{self.sample_id}_loop_centering_{omega_pos}_zoom_{md3.zoom.get()}",
            )
            self.save_image(
                data,
                x_coord,
                y_coord,
                filename,
            )

        return x_coord, y_coord

    def find_edge_and_flat_angles(self) -> Generator[Msg, None, None]:
        """
        Finds maximum and minimum area of a loop corresponding to the edge and
        flat angles of a loop by calculating. The data is then and normalized and
        fitted to a sine wave assuming that the period T of the sine function is known
        (T=pi by definition).
        During this step, we additionally run a three-point centering calculation to
        improve the loop-centering accuracy. If we find that the loop centering
        percentage error exceeds 2%, we stop the plan and set
        successful_centering=False


        Yields
        ------
        Generator[Msg, None, None]
            A bluesky generator
        """
        start_omega = md3.omega.position
        omega_list = (
            np.array([start_omega, start_omega + 90, start_omega + 180]) % 360
        )  # degrees
        area_list = []
        x_coords = []
        y_coords = []

        # We zoom in and increase the backlight intensity to improve accuracy
        yield from mv(md3.zoom, self.final_zoom_level)
        yield from mv(md3.backlight, 2)

        for omega in omega_list:
            if omega != start_omega:
                yield from mv(md3.omega, omega)

            image = get_image_from_md3_camera(np.uint8)
            edge_detection = LoopEdgeDetection(
                image,
                block_size=self.md3_cam_block_size,
                adaptive_constant=self.md3_cam_adaptive_constant,
            )

            extremes = edge_detection.find_extremes()

            x_coords.append(extremes.top[0] / md3.zoom.pixels_per_mm)
            y_coords.append(extremes.top[1] / md3.zoom.pixels_per_mm)

            if self.plot:
                filename = path.join(
                    self.sample_path,
                    f"{self.sample_id}_area_estimation_{round(md3.omega.position)}",
                )
                self.save_image(
                    image,
                    extremes.top[0],
                    extremes.top[1],
                    filename,
                )
            area_list.append(edge_detection.loop_area())

        yield from self.three_click_centering(
            x_coords, y_coords, np.radians(omega_list)
        )

        if BL_ACTIVE == "false":
            # Don't bother about statistics in simulation mode
            # since we only stream static images
            successful_centering = True
            self.flat_angle = 0
            self.edge_angle = 90
            logger.warning("BL_ACTIVE=False, centering statics will be ignored")
            return successful_centering

        x, y = self.find_loop_edge_coordinates()
        percentage_error_x = (
            abs((x - self.beam_position[0]) / self.beam_position[0]) * 100
        )
        percentage_error_y = (
            abs((y - self.beam_position[1]) / self.beam_position[1]) * 100
        )

        if (
            percentage_error_x > self.percentage_error
            or percentage_error_y > self.percentage_error
        ):
            successful_centering = False
            raise ValueError(
                "Optical loop centering has probably failed. The percentage errors "
                f"for the x and y axis are {self.percentage_error}% and "
                f"{percentage_error_y}% respectively. We only tolerate errors "
                "up to 7%."
            )
        else:
            successful_centering = True

        # Normalize the data (we do not care about amplitude,
        # we only care about phase)
        area_list = np.array(area_list)
        area_list = area_list / np.linalg.norm(area_list)

        # Fit the curve
        optimised_params, _ = optimize.curve_fit(
            self._sine_function,
            np.radians(omega_list),
            np.array(area_list),
            p0=[0.2, 0, 0.2],
            maxfev=4000,
            bounds=([0, -2 * np.pi, -np.inf], [1, 2 * np.pi, np.inf]),
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
            plt.scatter(np.radians(omega_list), np.array(area_list), label="Data")
            plt.xlabel("omega [radians]")
            plt.ylabel("Area [pixels^2]")
            plt.legend(fontsize=15)
            # plt.tight_layout()
            filename = path.join(self.sample_path, f"{self.sample_id}_area_curve_fit")
            plt.savefig(filename, dpi=70)
            plt.close()
        return successful_centering

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

    def _find_zoom_0_maximum_area(self) -> Generator[Msg, None, tuple[float, float]]:
        """
        Finds the angle where the area of the loop is maximum.
        This means that the tip of the loop at zoom level 0
        is calculated more accurately

        Yields
        ------
        Generator[Msg, None, NDArray]
            a bluesky plan that returns the coordinates of the tip of the loop
        """

        initial_omega = md3.omega.position
        omega_list = [initial_omega, initial_omega + 90]
        area_list = []
        tip_coordinates = []

        for omega in omega_list:
            if omega != initial_omega:
                yield from mv(md3.omega, omega)

            img, height, width = get_image_from_top_camera(np.uint8)
            img = img.reshape(height, width)
            img = img[
                self.top_camera_roi_y[0] : self.top_camera_roi_y[1],
                self.top_camera_roi_x[0] : self.top_camera_roi_x[1],
            ]

            edge_detection = LoopEdgeDetection(
                img,
                block_size=self.top_cam_block_size,
                adaptive_constant=self.top_cam_adaptive_constant,
            )
            area_list.append(edge_detection.loop_area())
            tip = edge_detection.find_tip()
            tip_coordinates.append(tip)

            if self.plot:
                filename = path.join(
                    self.sample_path, f"{self.sample_id}_{round(omega)}_top_camera"
                )
                self.save_image(
                    img,
                    tip[0],
                    tip[1],
                    filename,
                    grayscale_img=True,
                )

        argmax = np.argmax(area_list)
        yield from mv(md3.omega, omega_list[argmax])

        # average results for consistency
        x_coord = [tip_coordinates[argmax][0]]
        y_coord = [tip_coordinates[argmax][1]]
        for _ in range(5):
            img, height, width = get_image_from_top_camera(np.uint8)
            img = img.reshape(height, width)
            img = img[
                self.top_camera_roi_y[0] : self.top_camera_roi_y[1],
                self.top_camera_roi_x[0] : self.top_camera_roi_x[1],
            ]
            edge_detection = LoopEdgeDetection(
                img,
                block_size=self.top_cam_block_size,
                adaptive_constant=self.top_cam_adaptive_constant,
            )
            tip = edge_detection.find_tip()
            x_coord.append(tip[0])
            y_coord.append(tip[1])
        return (np.median(x_coord), np.median(y_coord))

    def _calculate_p_value(self, image: npt.NDArray):
        """Calculates the p value of the zoom level 0 image with
        the background image.

        Parameters
        ----------
        image : npt.NDArray
            The zoom level 0 image

        Raises
        ------
        RuntimeError
            Raises an error if the p_value is greater than 0.9,
            this indicates that a pin has most likely not been found
        """
        p_value = kstest(image, top_camera_background_img_array).pvalue
        # Check if there is a pin using the KS test
        if p_value > 0.9:
            raise ValueError(
                "No pin found during the pre-centering step. "
                "Optical and x-ray centering will not continue"
            )

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
        # Always use the same start position as reference
        start_sample_x = 0
        start_sample_y = 0
        start_omega = 0
        start_alignment_z = 0
        start_alignment_y = 0
        start_alignment_x = 0.434
        yield from md3_move(
            md3.omega,
            start_omega,
            md3.alignment_y,
            start_alignment_y,
            md3.sample_x,
            start_sample_x,
            md3.sample_y,
            start_sample_y,
            md3.alignment_z,
            start_alignment_z,
            md3.alignment_x,
            start_alignment_x,
        )

        if md3.zoom.position != 1:
            yield from mv(md3.zoom, 1)

        if md3.zoom.get() != 1:
            raise ValueError(
                "The MD3 zoom could not be changed. Check the MD3 UI and try again."
            )

        if round(md3.backlight.get()) != 2:
            yield from mv(md3.backlight, 2)

        screen_coordinates = yield from self._find_zoom_0_maximum_area()

        x_coord = screen_coordinates[0]
        y_coord = screen_coordinates[1]

        delta_mm_x = (self.x_pixel_target - x_coord) / self.top_cam_pixels_per_mm_x
        delta_mm_y = (self.y_pixel_target - y_coord) / self.top_cam_pixels_per_mm_y
        yield from md3_move(
            md3.alignment_y,
            md3.alignment_y.position - delta_mm_y,
            md3.alignment_z,
            md3.alignment_z.position - delta_mm_x,
        )

        return True

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
            s=150,
            c="r",
            marker="x",
        )
        plt.title(f"omega={round(md3.omega.position)}", fontsize=18)
        plt.savefig(filename, dpi=70)
        plt.close()

    def prepare_raster_grid(
        self, omega: float, filename: str = "step_3_prep_raster"
    ) -> RasterGridCoordinates:
        """
        Prepares a raster grid. The limits of the grid are obtained using
        the LoopEdgeDetection class

        Parameters
        ----------
        omega : float
            Angle at which the grid scan is done
        filename: str
            Name of the file used to save the results if self.plot = True,
            by default step_3_prep_raster

        Returns
        -------
        motor_coordinates: RasterGridCoordinates
            A pydantic model containing the initial and final motor positions of the grid,
            as well as its coordinates in units of pixels
        rectangle_coordinates: dict
            Rectangle coordinates in pixels
        """
        data = get_image_from_md3_camera(np.uint8)

        edge_detection = LoopEdgeDetection(
            data,
            block_size=self.md3_cam_block_size,
            adaptive_constant=self.md3_cam_adaptive_constant,
        )
        # TODO: determine experimentally the optimal value of
        # height_scale_factor
        rectangle_coordinates = edge_detection.fit_rectangle(
            height_scale_factor=self.grid_height_scale_factor
        )

        if self.plot:
            edge_detection.plot_raster_grid(
                rectangle_coordinates,
                filename,
            )

        width_pixels = abs(
            rectangle_coordinates.top_left[0] - rectangle_coordinates.bottom_right[0]
        )
        width_mm = width_pixels / md3.zoom.pixels_per_mm

        height_pixels = abs(
            rectangle_coordinates.top_left[1] - rectangle_coordinates.bottom_right[1]
        )
        height_mm = height_pixels / md3.zoom.pixels_per_mm

        # Y pixel coordinates
        initial_pos_y_pixels = rectangle_coordinates.top_left[1] - self.beam_position[1]
        final_pos_y_pixels = (
            rectangle_coordinates.bottom_right[1] - self.beam_position[1]
        )

        # Alignment y target positions (mm)
        initial_pos_alignment_y = (
            md3.alignment_y.position + initial_pos_y_pixels / md3.zoom.pixels_per_mm
        )
        final_pos_alignment_y = (
            md3.alignment_y.position + final_pos_y_pixels / md3.zoom.pixels_per_mm
        )

        # X pixel coordinates
        initial_pos_x_pixels = rectangle_coordinates.top_left[0] - self.beam_position[0]
        final_pos_x_pixels = (
            rectangle_coordinates.bottom_right[0] - self.beam_position[0]
        )

        # Sample x target positions (mm)
        initial_pos_sample_x = md3.sample_x.position + np.sin(
            np.radians(md3.omega.position)
        ) * (initial_pos_x_pixels / md3.zoom.pixels_per_mm)
        final_pos_sample_x = md3.sample_x.position + np.sin(
            np.radians(md3.omega.position)
        ) * (+final_pos_x_pixels / md3.zoom.pixels_per_mm)

        # Sample y target positions (mm)
        initial_pos_sample_y = md3.sample_y.position + np.cos(
            np.radians(md3.omega.position)
        ) * (initial_pos_x_pixels / md3.zoom.pixels_per_mm)
        final_pos_sample_y = md3.sample_y.position + np.cos(
            np.radians(md3.omega.position)
        ) * (final_pos_x_pixels / md3.zoom.pixels_per_mm)

        # Center of the grid (mm) (y-axis only)
        center_x_of_grid_pixels = (
            rectangle_coordinates.top_left[0] + rectangle_coordinates.bottom_right[0]
        ) / 2
        center_pos_sample_x = md3.sample_x.position + np.sin(
            np.radians(md3.omega.position)
        ) * ((center_x_of_grid_pixels - self.beam_position[0]) / md3.zoom.pixels_per_mm)
        center_pos_sample_y = md3.sample_y.position + np.cos(
            np.radians(md3.omega.position)
        ) * ((center_x_of_grid_pixels - self.beam_position[0]) / md3.zoom.pixels_per_mm)

        # NOTE: The width and height are measured in mm and the grid_step in micrometers,
        # hence the conversion below
        number_of_columns = int(np.ceil(width_mm / (self.grid_step[0] / 1000)))
        number_of_rows = int(np.ceil(height_mm / (self.grid_step[1] / 1000)))

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
            number_of_columns=number_of_columns,
            number_of_rows=number_of_rows,
            top_left_pixel_coordinates=tuple(rectangle_coordinates.top_left),
            bottom_right_pixel_coordinates=tuple(rectangle_coordinates.bottom_right),
            width_pixels=width_pixels,
            height_pixels=height_pixels,
            md3_camera_pixel_width=self.md3_camera.width.get(),
            md3_camera_pixel_height=self.md3_camera.height.get(),
            md3_camera_snapshot=self._get_md3_camera_jpeg_image(),
            pixels_per_mm=md3.zoom.pixels_per_mm,
        )

        return raster_grid_coordinates

    def _get_md3_camera_jpeg_image(self) -> bytes:
        """
        Gets a numpy array from the md3 camera and stores it as a JPEG image
        using the io and PIL libraries

        Returns
        -------
        bytes
            The md3 camera image in bytes format
        """
        array = get_image_from_md3_camera("uint8")
        pil_image = Image.fromarray(array)

        with BytesIO() as f:
            pil_image.save(f, format="JPEG")
            jpeg_image = f.getvalue()

        return jpeg_image
