import logging
from typing import Generator

import cv2
import lucid3
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from bluesky.plan_stubs import mv
from bluesky.utils import Msg
from ophyd.signal import ConnectionTimeoutError

from mx3_beamline_library.devices.classes.detectors import BlackFlyCam
from mx3_beamline_library.devices.classes.motors import CosylabMotor
from mx3_beamline_library.science.optical_and_loop_centering.psi_optical_centering import (
    loopImageProcessing,
)

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)


class OpticalCentering:
    """
    This class runs a bluesky plan that optically centers the loop
    using the loop centering code developed by PSI. Before analysing an image,
    we can optionally unblur the image at the start of the plan to make sure the
    results are consistent.
    """

    def __init__(
        self,
        camera: BlackFlyCam,
        motor_x: CosylabMotor,
        motor_y: CosylabMotor,
        motor_z: CosylabMotor,
        motor_phi: CosylabMotor,
        beam_position: tuple[int, int],
        pixels_per_mm_x: float,
        pixels_per_mm_z: float,
        auto_focus: bool = True,
        min_focus: float = 0.0,
        max_focus: float = 1.0,
        tol: float = 0.3,
        method: str = "psi",
        plot: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        camera : BlackFlyCam
            Camera
        motor_x : CosylabMotor
            Motor X
        motor_y : CosylabMotor
            Motor Y
        motor_z : CosylabMotor
            Motor Z
        motor_phi : CosylabMotor
            Omega
        beam_position : tuple[int, int]
            Position of the beam
        pixels_per_mm_x : float
            Pixels per mm x
        pixels_per_mm_z : float
            Pixels per mm z
        auto_focus : bool, optional
            If true, we autofocus the image once before running the loop centering,
            algorithm, by default True
        min_focus : float, optional
            Minimum value to search for the maximum of var( Img * L(x,y) ),
            by default 0.0
        max_focus : float, optional
            Maximum value to search for the maximum of var( Img * L(x,y) ),
            by default 1.0
        tol : float, optional
            The tolerance used by the Golden-section search, by default 0.3
        method : str, optional
            Method used to find the edge of the loop. Can be either
            psi or lucid, by default "psi"
        plot : bool, optional
            If true, we take snapshots of the loop at different stages
            of the plan, by default False

        Returns
        -------
        None
        """
        self.camera = camera
        self.motor_x = motor_x
        self.motor_y = motor_y
        self.motor_z = motor_z
        self.motor_phi = motor_phi
        self.beam_position = beam_position
        self.pixels_per_mm_x = pixels_per_mm_x
        self.pixels_per_mm_z = pixels_per_mm_z
        self.auto_focus = auto_focus
        self.min_focus = min_focus
        self.max_focus = max_focus
        self.tol = tol
        self.method = method
        self.plot = plot

    def center_loop(self) -> Generator[Msg, None, None]:
        """
        This plan is the main optical loop centering plan, which is used by the
        optical_and_xray_centering plan. Here, we optically center the loop using the loop
        centering code developed by PSI. Before analysing an image, we can optionally unblur the
        image at the start of the plan to make sure the results are consistent.

        Yields
        ------
        Generator[Msg, None, None]
            A plan that automatically centers a loop
        """

        omega_list = [0, 90, 180]
        for omega in omega_list:
            yield from mv(self.motor_phi, omega)
            logger.info(f"Omega: {self.motor_phi.position}")

            if self.auto_focus and omega == 0:
                yield from self.unblur_image(self.min_focus, self.max_focus, self.tol)

            yield from self.drive_motors_to_loop_edge()

        # yield from self.drive_motors_to_center_of_loop()

    def unblur_image(
        self,
        a: float = 0.0,
        b: float = 1.0,
        tol: float = 0.2,
    ) -> float:
        """
        We use the Golden-section search to find the maximum of the variance function described
        in the calculate_variance method ( `var( Img * L(x,y) )` ). We assume that the function
        is strictly unimodal on [a,b].
        See for example: https://en.wikipedia.org/wiki/Golden-section_search

        Parameters
        ----------
        a : float
            Minimum value to search for the maximum of var( Img * L(x,y) )
        b : float
            Maximum value to search for the maximum of var( Img * L(x,y) )
        tol : float, optional
            The tolerance, by default 0.2

        Returns
        -------
        Generator[Msg, None, None]
            Moves motor_y to a position where the image is focused
        """
        gr = (np.sqrt(5) + 1) / 2

        c = b - (b - a) / gr
        d = a + (b - a) / gr

        count = 0
        logger.info("Focusing image...")
        while abs(b - a) > tol:
            yield from mv(self.motor_y, c)
            val_c = self.calculate_variance()

            yield from mv(self.motor_y, d)
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
        logger.info(f"Optimal motor_y value: {maximum}")
        yield from mv(self.motor_y, maximum)

    def drive_motors_to_loop_edge(self) -> Generator[Msg, None, None]:
        """
        Drives motor_x and motor_z to the edge of the loop. The edge of the loop is found
        using either Lucid3 of the PSI loop centering code

        Raises
        ------
        NotImplementedError
            An error if method is not lucid3 or psi

        Yields
        ------
        Generator[Msg, None, None]
            A message that tells bluesky to move the motors to the edge of the loop
        """
        data = self.get_image_from_camera(np.uint8)
        if self.method.lower() == "lucid3":
            loop_detected, x_coord, y_coord = lucid3.find_loop(
                image=data,
                rotation=True,
                rotation_k=2,
            )
        elif self.method.lower() == "psi":
            procImg = loopImageProcessing(data)
            procImg.findContour(zoom="-208.0", beamline="X06DA")
            extremes = procImg.findExtremes()
            screen_coordinates = extremes["bottom"]
            x_coord = screen_coordinates[0]
            y_coord = screen_coordinates[1]
        else:
            raise NotImplementedError(
                f"Supported methods are lucid3 and psi, not {self.method}"
            )

        logger.info(f"screen coordinates: {screen_coordinates}")

        if self.plot:
            self.save_image(
                data,
                x_coord,
                y_coord,
                f"step_2_loop_centering_fig_{x_coord}",
            )

        loop_position_x = (
            self.motor_x.position
            + (x_coord - self.beam_position[0]) / self.pixels_per_mm_x
        )
        loop_position_z = (
            self.motor_z.position
            + (y_coord - self.beam_position[1]) / self.pixels_per_mm_z
        )
        yield from mv(self.motor_x, loop_position_x, self.motor_z, loop_position_z)

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

        data = self.get_image_from_camera(np.uint8)

        procImg = loopImageProcessing(data)
        procImg.findContour(zoom="-208.0", beamline="X06DA")
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
                "step_2_centered_loop",
            )

        loop_position_x = (
            self.motor_x.position
            + (pos_x_pixels - self.beam_position[0]) / self.pixels_per_mm_x
        )
        loop_position_z = (
            self.motor_z.position
            + (pos_z_pixels - self.beam_position[1]) / self.pixels_per_mm_z
        )
        yield from mv(self.motor_x, loop_position_x, self.motor_z, loop_position_z)

    def calculate_variance(self) -> float:
        """
        We calculate the variance of the convolution of the laplacian kernel with an image,
        e.g. var( Img * L(x,y) ), where Img is an image taken from the camera ophyd object,
        and L(x,y) is the Laplacian kernel.

        Parameters
        ----------
        camera : BlackFlyCam
            A camera ophyd object

        Returns
        -------
        float
            var( Img * L(x,y) )
        """
        data = self.get_image_from_camera()

        gray_image = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)

        return cv2.Laplacian(gray_image, cv2.CV_64F).var()

    def get_image_from_camera(self, dtype: npt.DTypeLike = np.uint16) -> npt.NDArray:
        """
        Gets a frame from the camera an reshapes it as
        (height, width, depth).

        Parameters
        ----------
        dtype : npt.DTypeLike, optional
            The data type of the numpy array, by default np.uint16

        Returns
        -------
        npt.NDArray
            A frame of shape (height, width, depth)
        """
        try:
            array_data: npt.NDArray = self.camera.array_data.get()
            data = array_data.reshape(
                self.camera.height.get(),
                self.camera.width.get(),
                self.camera.depth.get(),
            ).astype(dtype)
        except ConnectionTimeoutError:
            # When the camera is not working, we stream a static image
            # of the test rig
            data = np.load("/mnt/shares/smd_share/blackfly_cam_images/flat.py").astype(
                dtype
            )

        return data

    def save_image(
        self, data: npt.NDArray, x_coord: float, y_coord: float, filename: str
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
        filename : str
            The filename

        Returns
        -------
        None
        """
        plt.figure()
        plt.imshow(data)
        plt.scatter(x_coord, y_coord, s=200, c="r", marker="+")
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
        data = self.get_image_from_camera()
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
