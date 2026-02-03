from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from bluesky.plan_stubs import mv
from bluesky.tracing import trace_plan, tracer
from bluesky.utils import Msg

from ...config import BL_ACTIVE, redis_connection
from ...devices.detectors import blackfly_camera
from ...devices.motors import md3
from ...logger import setup_logger
from ...schemas.optical_centering import OpticalCenteringExtraConfig, TopCameraConfig
from ...science.optical_and_loop_centering.loop_edge_detection import LoopEdgeDetection
from ..image_analysis import get_image_from_top_camera

logger = setup_logger(__name__)


class TopCameraTargetCoords:
    """
    This class is used to set the top camera target coordinates in redis.
    This assumes that the sample has been previously manually centered.
    """

    def __init__(
        self,
        plot: bool = False,
        extra_config: OpticalCenteringExtraConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------
        plot : bool, optional
            If True, the images taken by the top camera are plotted and saved, by default False
        extra_config : OpticalCenteringExtraConfig | None, optional
            The extra configuration values used during optical centering. If None, the default
            value is set to OpticalCenteringExtraConfig()
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

        self.top_cam_block_size = (
            optical_centering_config.top_camera.loop_image_processing.block_size
        )
        self.top_cam_adaptive_constant = (
            optical_centering_config.top_camera.loop_image_processing.adaptive_constant
        )
        self.top_camera_roi_x = optical_centering_config.top_camera.roi_x
        self.top_camera_roi_y = optical_centering_config.top_camera.roi_y

    def _calculate_target_coords(self) -> Generator[Msg, None, tuple[float, float]]:
        """
        Finds the angle where the area of the loop is maximum.
        This means that the tip of the loop at zoom level 0
        is calculated more accurately. Then, the tip of the loop
        is calculated 6 times at that angle and averaged.

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

    @trace_plan(tracer, "set_top_camera_target_coords")
    def set_top_camera_target_coords(self) -> Generator[Msg, None, None]:
        """
        Sets the top camera target coordinates in redis for the top camera
        under the key "top_camera_target_coords".
        This assumes that the sample has been previously manually centered.

        Yields
        ------
        Generator[Msg, None, None]
            A bluesky generator
        """
        if md3.zoom.position != 1:
            yield from mv(md3.zoom, 1)

        if md3.zoom.get() != 1:
            raise ValueError(
                "The MD3 zoom could not be changed. Check the MD3 UI and try again."
            )

        if round(md3.backlight.get()) != 2:
            yield from mv(md3.backlight, 2)

        screen_coordinates = yield from self._calculate_target_coords()

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
