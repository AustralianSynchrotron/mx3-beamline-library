from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from bluesky.plan_stubs import mv
from bluesky.utils import Msg

from ...config import BL_ACTIVE, redis_connection
from ...devices.detectors import blackfly_camera
from ...devices.motors import md3
from ...logger import setup_logger
from ...schemas.optical_centering import OpticalCenteringExtraConfig, TopCameraConfig
from ...science.optical_and_loop_centering.loop_edge_detection import LoopEdgeDetection
from ..image_analysis import get_image_from_top_camera
from ..plan_stubs import md3_move

logger = setup_logger()


class TopCameraTargetCoords:
    """
    This class runs a bluesky plan that optically aligns the loop with the
    center of the beam. Before analysing an image, we unblur the image at the start
    of the plan to make sure the results are consistent. Finally we find angles at which
    the area of a loop is maximum and minimum (flat and edge) and we calculate the grid
    coordinates for the flat and edge angles.
    """

    def __init__(
        self,
        plot: bool = False,
        extra_config: OpticalCenteringExtraConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------
        sample_id : str
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
        self.top_camera = blackfly_camera
        self.plot = plot

        self._check_top_camera_config()
        self._set_optical_centering_config_parameters(extra_config)

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
        self.percentage_error = (
            optical_centering_config.optical_centering_percentage_error
        )

    def center_loop(self) -> Generator[Msg, None, None]:
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
            yield from self.set_top_camera_target_coords()
        else:
            pass

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
                filename = f"top_camera_omega_{int(omega)}"
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

    def set_top_camera_target_coords(self) -> Generator[Msg, None, None]:
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
        start_alignment_y = 0
        start_sample_x = 0
        start_sample_y = 0
        start_omega = 0
        start_alignment_z = 0
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

        logger.info(f"x_pixel_target: {x_coord}")
        logger.info(f"y_pixel_target: {y_coord}")

        redis_connection.hset(
            "top_camera_target_coords",
            mapping={
                "x_pixel_target": float(x_coord),
                "y_pixel_target": float(y_coord),
            },
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
