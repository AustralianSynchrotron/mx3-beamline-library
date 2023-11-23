import logging
import pickle
from io import BytesIO
from os import environ, getcwd, mkdir, path
from typing import Generator, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import redis
import yaml
from bluesky.preprocessors import run_wrapper
from bluesky.utils import Msg
from matplotlib import rc
from ophyd import Signal
from PIL import Image
from scipy import optimize
from scipy.stats import kstest

from ..config import BL_ACTIVE
from ..constants import top_camera_background_img_array
from ..devices.detectors import blackfly_camera, md3_camera
from ..devices.motors import md3
from ..schemas.optical_centering import (
    CenteredLoopMotorCoordinates,
    OpticalCenteringResults,
)
from ..schemas.xray_centering import RasterGridCoordinates
from ..science.optical_and_loop_centering.loop_edge_detection import LoopEdgeDetection
from .image_analysis import (
    get_image_from_md3_camera,
    get_image_from_top_camera,
    unblur_image_fast,
)
from .plan_stubs import md3_move, move_and_emit_document as mv

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)


rc("xtick", labelsize=15)
rc("ytick", labelsize=15)
# Set usetex=True to get nice LATEX plots. Note that
# using LATEX significantly reduces speed!
rc("text", usetex=False)

path_to_config_file = path.join(
    path.dirname(__file__), "configuration/optical_and_xray_centering.yml"
)

with open(path_to_config_file, "r") as plan_config:
    PLAN_CONFIG: dict = yaml.safe_load(plan_config)


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
        sample_id: str,
        beam_position: tuple[int, int],
        grid_step: Union[tuple[float, float], None] = None,
        calibrated_alignment_z: float = 0.634,
        auto_focus: bool = True,
        min_focus: float = -0.3,
        max_focus: float = 1.3,
        tol: float = 0.3,
        number_of_intervals: int = 2,
        plot: bool = False,
        x_pixel_target: int = 841,
        y_pixel_target: int = 472,
        top_camera_background_img_array: npt.NDArray = None,
        top_camera_roi_x: tuple[int, int] = (0, 1224),
        top_camera_roi_y: tuple[int, int] = (100, 1024),
        output_directory: Union[str, None] = None,
        use_top_camera_camera: bool = True,
        manual_mode: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        sample_id : str
            Sample id
        beam_position : tuple[int, int]
            Position of the beam
        grid_step : Union[tuple[float, float], None]
            The step of the grid (x,y) in micrometers. Can also be None
            only if manual_mode=True
        calibrated_alignment_z : float, optional.
            The alignment_z position which aligns a sample with the center of rotation
            at the beam position. This value is calculated experimentally, by default
            0.662
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
        top_camera_roi_x : tuple[int, int], optional
            X Top camera region of interest, by default (0, 1224)
        top_camera_roi_y : tuple[int, int], optional
            Y Top camera region of interest, by default (100, 1024)
        output_directory : Union[str, None], optional
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

        Returns
        -------
        None
        """
        self.sample_id = sample_id
        self.md3_camera = md3_camera
        self.top_camera = blackfly_camera
        self.sample_x = md3.sample_x
        self.sample_y = md3.sample_y
        self.alignment_x = md3.alignment_x
        self.alignment_y = md3.alignment_y
        self.alignment_z = md3.alignment_z
        self.omega = md3.omega
        self.zoom = md3.zoom
        self.phase = md3.phase
        self.backlight = md3.backlight
        self.beam_position = beam_position
        self.grid_step = grid_step
        self.auto_focus = auto_focus
        self.min_focus = min_focus
        self.max_focus = max_focus
        self.tol = tol
        self.number_of_intervals = number_of_intervals
        self.plot = plot
        self.x_pixel_target = x_pixel_target
        self.y_pixel_target = y_pixel_target
        self.top_camera_roi_x = top_camera_roi_x
        self.top_camera_roi_y = top_camera_roi_y
        self.calibrated_alignment_z = calibrated_alignment_z

        self.centered_loop_coordinates = None

        self.md3_cam_block_size = PLAN_CONFIG["loop_image_processing"]["md3_camera"][
            "block_size"
        ]
        self.md3_cam_adaptive_constant = PLAN_CONFIG["loop_image_processing"][
            "md3_camera"
        ]["adaptive_constant"]
        self.top_cam_block_size = PLAN_CONFIG["loop_image_processing"]["top_camera"][
            "block_size"
        ]
        self.top_cam_adaptive_constant = PLAN_CONFIG["loop_image_processing"][
            "top_camera"
        ]["adaptive_constant"]
        self.alignment_x_default_pos = PLAN_CONFIG["motor_default_positions"][
            "alignment_x"
        ]

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

        if output_directory is None:
            self.output_directory = getcwd()
        else:
            self.output_directory = output_directory

        self.sample_path = path.join(self.output_directory, self.sample_id)
        if self.plot:
            try:
                mkdir(self.sample_path)
            except FileExistsError:
                pass
        self.use_top_camera_camera = use_top_camera_camera
        self.manual_mode = manual_mode

        if not self.manual_mode:
            assert (
                self.grid_step is not None
            ), "grid_step can only be None if manual_mode=True"

    def center_loop(self) -> Generator[Msg, None, None]:
        """
        Opens and closes the run while keeping track of the signals
        used in the loop centering plans

        Yields
        ------
        Generator[Msg, None, None]
            The loop centering plan generator
        """
        yield from run_wrapper(self._center_loop(), md={"sample_id": self.sample_id})

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
        current_phase = self.phase.get()
        if current_phase != "Centring":
            yield from mv(self.phase, "Centring")

        if self.use_top_camera_camera:
            loop_found = yield from self.move_loop_to_md3_field_of_view()
        else:
            loop_found = True

        if not loop_found:
            logger.error("No loop found by the zoom level-0 camera")
            optical_centering_results = OpticalCenteringResults(
                optical_centering_successful=False
            )
            self.redis_connection.set(
                f"optical_centering_results:{self.sample_id}",
                pickle.dumps(optical_centering_results.dict()),
            )
            return

        # We center the loop at two different zooms
        yield from mv(self.zoom, 1)
        yield from self.multi_point_centering_plan()

        successful_centering = yield from self.find_edge_and_flat_angles()
        self.centered_loop_coordinates = CenteredLoopMotorCoordinates(
            alignment_x=self.alignment_x.position,
            alignment_y=self.alignment_y.position,
            alignment_z=self.alignment_z.position,
            sample_x=self.sample_x.position,
            sample_y=self.sample_y.position,
        )
        if not self.manual_mode:
            if not successful_centering:
                optical_centering_results = OpticalCenteringResults(
                    optical_centering_successful=False
                )
                self.redis_connection.set(
                    f"optical_centering_results:{self.sample_id}",
                    pickle.dumps(optical_centering_results.dict()),
                )
                return

            # Prepare grid for the edge surface
            yield from mv(self.zoom, 4)
            yield from mv(self.omega, self.edge_angle)
            filename_edge = path.join(
                self.sample_path, f"{self.sample_id}_raster_grid_edge"
            )
            grid_edge = self.prepare_raster_grid(self.edge_angle, filename_edge)
            # Add metadata for bluesky documents
            mv(self.grid_scan_coordinates_edge, grid_edge.dict())

            # Prepare grid for the flat surface
            yield from mv(self.zoom, 4)
            yield from mv(self.omega, self.flat_angle)
            filename_flat = path.join(
                self.sample_path, f"{self.sample_id}_raster_grid_flat"
            )
            grid_flat = self.prepare_raster_grid(self.flat_angle, filename_flat)
            # Add metadata for bluesky documents
            mv(self.grid_scan_coordinates_flat, grid_flat.dict())

            optical_centering_results = OpticalCenteringResults(
                optical_centering_successful=True,
                centered_loop_coordinates=self.centered_loop_coordinates,
                edge_angle=self.edge_angle,
                flat_angle=self.flat_angle,
                edge_grid_motor_coordinates=grid_edge,
                flat_grid_motor_coordinates=grid_flat,
            )

            # Save results to redis
            self.redis_connection.set(
                f"optical_centering_results:{self.sample_id}",
                pickle.dumps(optical_centering_results.dict()),
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
        yield from mv(self.zoom, 1)
        start_omega = self.omega.position
        omega_array = np.array([start_omega, start_omega + 90]) % 360
        focused_position_list = []
        for i, omega in enumerate(omega_array):
            yield from mv(self.omega, omega)
            if self.auto_focus:
                if i % 2:
                    focused_alignment_x = yield from unblur_image_fast(
                        self.alignment_x,
                        start_position=self.min_focus,
                        final_position=self.max_focus,
                    )
                    focused_position_list.append(focused_alignment_x)
                else:
                    focused_alignment_x = yield from unblur_image_fast(
                        self.alignment_x,
                        start_position=self.max_focus,
                        final_position=self.min_focus,
                    )
                    focused_position_list.append(focused_alignment_x)

        # Start from the last positions (this optimises the plan)
        omega_array = np.flip(omega_array)
        focused_position_list.reverse()

        x_coords, y_coords = [], []
        for omega, alignment_x in zip(omega_array, focused_position_list):
            yield from md3_move(self.omega, omega, self.alignment_x, alignment_x)

            x, y = self.find_loop_edge_coordinates()
            x_coords.append(x / self.zoom.pixels_per_mm)
            y_coords.append(y / self.zoom.pixels_per_mm)

        yield from self.drive_motors_to_aligned_position(
            x_coords, y_coords, np.radians(omega_array)
        )

    def drive_motors_to_aligned_position(
        self, x_coords: list, y_coords: list, omega_positions: list
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
        average_y_position = np.mean(y_coords)

        amplitude, phase = self.multi_point_centre(x_coords, omega_positions)
        delta_sample_y = amplitude * np.sin(phase)
        delta_sample_x = amplitude * np.cos(phase)

        delta_alignment_y = average_y_position - (
            self.beam_position[1] / self.zoom.pixels_per_mm
        )

        yield from md3_move(
            self.sample_x,
            self.sample_x.position + delta_sample_x,
            self.sample_y,
            self.sample_y.position + delta_sample_y,
            self.alignment_y,
            self.alignment_y.position + delta_alignment_y,
            self.alignment_z,
            self.calibrated_alignment_z,
            self.alignment_x,
            self.alignment_x_default_pos,
        )

    def multi_point_centre(self, x_coords: list, omega_list: list) -> npt.NDArray:
        """
        Multipoint centre function

        Parameters
        ----------
        x_coords : list
            A list of x-coordinates values obtained during
            three-click centering in mm
        omega_list : list
            A list containing a list of omega values in radians

        Returns
        -------
        npt.NDArray
            The optimised parameters: (amplitude, phase)
        """

        optimised_params, _ = optimize.curve_fit(
            self._centering_function, omega_list, x_coords, p0=[1.0, 0.0]
        )

        return optimised_params

    def _centering_function(
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
        offset = self.beam_position[0] / self.zoom.pixels_per_mm
        return amplitude * np.sin(theta + phase) + offset

    def find_loop_edge_coordinates(self) -> tuple[float, float]:
        """
        We find the edge of the loop using the LoopEdgeDetection class.

        Returns
        -------
        tuple[float, float]
            The x and y pixel coordinates of the edge of the loop,
        """
        data = get_image_from_md3_camera(np.uint8)

        edge_detection = LoopEdgeDetection(
            data,
            block_size=self.md3_cam_block_size,
            adaptive_constant=self.md3_cam_adaptive_constant,
        )
        screen_coordinates = edge_detection.find_tip()

        x_coord = screen_coordinates[0]
        y_coord = screen_coordinates[1]

        if self.plot:
            omega_pos = round(self.omega.position)
            filename = path.join(
                self.sample_path,
                f"{self.sample_id}_loop_centering_{omega_pos}_zoom_{self.zoom.get()}",
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
        start_omega = self.omega.position
        omega_list = (
            np.array([start_omega, start_omega + 90, start_omega + 180]) % 360
        )  # degrees
        area_list = []
        x_coords = []
        y_coords = []

        # We zoom in and increase the backlight intensity to improve accuracy
        yield from mv(self.zoom, 4)
        yield from mv(self.backlight, 2)

        for omega in omega_list:
            yield from mv(self.omega, omega)

            image = get_image_from_md3_camera(np.uint8)
            edge_detection = LoopEdgeDetection(
                image,
                block_size=self.md3_cam_block_size,
                adaptive_constant=self.md3_cam_adaptive_constant,
            )

            extremes = edge_detection.find_extremes()

            x_coords.append(extremes.top[0] / self.zoom.pixels_per_mm)
            y_coords.append(extremes.top[1] / self.zoom.pixels_per_mm)

            if self.plot:
                filename = path.join(
                    self.sample_path,
                    f"{self.sample_id}_area_estimation_{round(self.omega.position)}",
                )
                self.save_image(
                    image,
                    extremes.top[0],
                    extremes.top[1],
                    filename,
                )
            area_list.append(edge_detection.loop_area())

        yield from self.drive_motors_to_aligned_position(
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

        if percentage_error_x > 2 or percentage_error_y > 2:
            successful_centering = False
            logger.error(
                "Optical loop centering has probably failed. The percentage errors "
                f"for the x and y axis are {percentage_error_x}% and "
                f"{percentage_error_y}% respectively. We only tolerate errors "
                "up to 1%."
            )
            return successful_centering
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
            plt.xlabel("$\omega$ [radians]", fontsize=18)
            plt.ylabel("Area [pixels$^2$]", fontsize=18)
            plt.legend(fontsize=15)
            plt.tight_layout()
            filename = path.join(self.sample_path, f"{self.sample_id}_area_curve_fit")
            plt.savefig(filename)
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
        if round(self.omega.position) != 0:
            yield from mv(self.omega, 0)
        if self.zoom.position != 1:
            yield from mv(self.zoom, 1)
        if round(self.backlight.get()) != 2:
            yield from mv(self.backlight, 2)

        img, height, width = get_image_from_top_camera(np.uint8)

        p_value = kstest(img, top_camera_background_img_array).pvalue
        # Check if there is a pin using the KS test
        if p_value > 0.9:
            logger.error(
                "No pin found during the pre-centering step. "
                "Optical and x-ray centering will not continue"
            )
            loop_found = False
            return loop_found
        else:
            loop_found = True

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
        screen_coordinates = edge_detection.find_tip()

        x_coord = screen_coordinates[0]
        y_coord = screen_coordinates[1]
        if self.plot:
            filename = path.join(self.sample_path, f"{self.sample_id}_top_camera")
            self.save_image(
                img,
                x_coord,
                y_coord,
                filename,
                grayscale_img=True,
            )

        delta_mm_x = (self.x_pixel_target - x_coord) / self.top_camera.pixels_per_mm_x
        delta_mm_y = (self.y_pixel_target - y_coord) / self.top_camera.pixels_per_mm_y
        yield from md3_move(
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
            s=150,
            c="r",
            marker="x",
        )
        plt.title(f"$\omega={round(self.omega.position)}^\circ$", fontsize=18)
        plt.savefig(filename)
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
        rectangle_coordinates = edge_detection.fit_rectangle()

        if self.plot:
            edge_detection.plot_raster_grid(
                rectangle_coordinates,
                filename,
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
            self.alignment_y.position - initial_pos_y_pixels / self.zoom.pixels_per_mm
        )
        final_pos_alignment_y = (
            self.alignment_y.position + final_pos_y_pixels / self.zoom.pixels_per_mm
        )

        # X pixel coordinates
        initial_pos_x_pixels = abs(
            rectangle_coordinates.top_left[0] - self.beam_position[0]
        )
        final_pos_x_pixels = abs(
            rectangle_coordinates.bottom_right[0] - self.beam_position[0]
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
            rectangle_coordinates.top_left[0] + rectangle_coordinates.bottom_right[0]
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
            initial_pos_alignment_z=self.alignment_z.position,
            final_pos_alignment_z=self.alignment_z.position,
            omega=omega,
            alignment_x_pos=self.alignment_x.position,
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
            pixels_per_mm=self.zoom.pixels_per_mm,
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
