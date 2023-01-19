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
from scipy import optimize

from mx3_beamline_library.devices.classes.detectors import BlackFlyCam
from mx3_beamline_library.devices.classes.motors import CosylabMotor, MD3Zoom, MD3Motor
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
        sample_x: CosylabMotor,
        sample_y: CosylabMotor,
        alignment_x,
        alignment_y: CosylabMotor,
        alignment_z: CosylabMotor,
        motor_phi: CosylabMotor,
        zoom: MD3Zoom,
        beam_position: tuple[int, int],
        pixels_per_mm_x: float,
        pixels_per_mm_z: float,
        auto_focus: bool = True,
        min_focus: float = 0.0,
        max_focus: float = 1.0,
        tol: float = 0.3,
        number_of_intervals: int = 1,
        method: str = "psi",
        plot: bool = False,
        loop_img_processing_beamline: str = "testrig",
        loop_img_processing_zoom: str = "1.0",
    ) -> None:
        """
        Parameters
        ----------
        camera : BlackFlyCam
            Camera
        sample_x : CosylabMotor
            Motor X
        sample_y : CosylabMotor
            Motor Y
        alignment_y : CosylabMotor
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
        loop_img_processing_beamline : str
            This name is used to get the configuration parameters used by the
            loop image processing code developed by PSI, by default testrig
        loop_img_processing_zoom : str
            We get the configuration parameters used by the loop image processing code
            for a particular zoom, by default 1.0

        Returns
        -------
        None
        """
        self.camera = camera
        self.sample_x = sample_x
        self.sample_y = sample_y
        self.alignment_x = alignment_x
        self.alignment_y = alignment_y
        self.alignment_z = alignment_z
        self.motor_phi = motor_phi
        self.zoom = zoom
        self.beam_position = beam_position
        self.pixels_per_mm_x = pixels_per_mm_x
        self.pixels_per_mm_z = pixels_per_mm_z
        self.auto_focus = auto_focus
        self.min_focus = min_focus
        self.max_focus = max_focus
        self.tol = tol
        self.method = method
        self.plot = plot
        self.loop_img_processing_beamline = loop_img_processing_beamline
        self.loop_img_processing_zoom = loop_img_processing_zoom
        self.current_focus_value = None
        self.number_of_intervals = number_of_intervals


    def center_loop(self):
        X, Y, phi_positions = [], [], []

        yield from mv(
            self.alignment_x, 0.434, self.alignment_y, 0, self.alignment_z,  0.63,
            self.sample_x, 0.2, self.sample_y, 0.3, self.motor_phi, 0, self.zoom, 1)

       
        # yield from self.drive_motors_to_loop_edge()
        # yield from mv(self.zoom, 3)
        
        omega_list = [0, 90 ,180]
        for omega in omega_list:
            yield from mv(self.motor_phi, omega)
            if self.auto_focus:
                yield from self.unblur_image(self.alignment_x, self.min_focus, self.max_focus, self.tol, self.number_of_intervals)
            #elif self.auto_focus and omega == 90:
            #    yield from self.unblur_image(self.alignment_x, self.min_focus, self.max_focus, self.tol, self.number_of_intervals)  
            

            x, y = self.find_loop_edge_coordinates()

            X.append(x / self.zoom.pixels_per_mm)
            Y.append(y / self.zoom.pixels_per_mm)
            phi_positions.append(np.radians(self.motor_phi.position))


        chi_angle = np.radians(90)
        chiRotMatrix = np.matrix(
            [
                [np.cos(chi_angle), -np.sin(chi_angle)],
                [np.sin(chi_angle), np.cos(chi_angle)],
            ]
        )
        Z = chiRotMatrix * np.matrix([X, Y])
        z = Z[1]
        avg_pos = Z[0].mean()

        r, a, offset = self.multiPointCentre(np.array(z).flatten(), phi_positions)
        dy = r * np.sin(a)
        dx = r * np.cos(a)

        d = chiRotMatrix.transpose() * np.matrix([[avg_pos], [offset]])

        d_horizontal = d[0] - (self.beam_position[0] / self.zoom.pixels_per_mm)
        d_vertical = d[1] - (self.beam_position[1] / self.zoom.pixels_per_mm)

        # NOTE: We drive alignment x to 0.434 as it corresponds to a
        # focused sample on the MD3
        yield from mv(
            self.sample_x, self.sample_x.position + dx,
            self.sample_y, self.sample_y.position + dy,
            self.alignment_y, self.alignment_y.position + d_vertical[0, 0],
            self.alignment_z, self.alignment_z.position - d_horizontal[0, 0],
            self.alignment_x, 0.434
            )


    def multiPointCentre(self, z, phis):
        def fitfunc(p, x):
            return p[0] * np.sin(x + p[1]) + p[2]

        def errfunc(p, x, y):
            return fitfunc(p, x) - y

        # The function call returns tuples of varying length
        result = optimize.leastsq(errfunc, [1.0, 0.0, 0.0], args=(phis, z))
        return result[0]

    def variance_local_maximum(
        self,
        focus_motor: MD3Motor,
        a: float = 0.0,
        b: float = 1.0,
        tol: float = 0.2,
    ) -> float:
        """
        We use the Golden-section search to find the local maximum of the variance function described
        in the calculate_variance method ( `var( Img * L(x,y) )` ). We assume that the function
        is strictly unimodal on [a,b].
        See for example: https://en.wikipedia.org/wiki/Golden-section_search

        Parameters
        ----------
        focus_motor : MD3Motor
            An MD3 motor
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
        number_of_intervals: int = 1,
    ):
        """
        We use the Golden-section search to find the global maximum of the variance function described
        in the calculate_variance method ( `var( Img * L(x,y) )` ) (see the definition of self.variance_local_maximum).
        In order to find the global maximum, we search for local maximums in N number of sub-intervals defined by
        number_of_intervals.

        Parameters
        ----------
        motor : MD3Motor
            An MD3 motor. We can focus the image with either alignment x, or sample_x and sample_y (depending on
            the value of omega)
        a : float
            Minimum value to search for the maximum of var( Img * L(x,y) )
        b : float
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
        print(step)
        interval_list = []
        for i in range(number_of_intervals):
            interval_list.append((a+step*i, a+step*(i+1)))

        # Calculate local maximum values
        laplacian_list = []
        focus_motor_pos_list = []
        for interval in interval_list:
            yield from self.variance_local_maximum(focus_motor, interval[0], interval[1], tol)
            laplacian_list.append(self.calculate_variance())
            focus_motor_pos_list.append(focus_motor.position)

        # Find global maximum, and move the focus motor to the best focused position
        argmax = np.argmax(np.array(laplacian_list))
        yield from mv(focus_motor, focus_motor_pos_list[argmax])


    def drive_motors_to_loop_edge(self) -> Generator[Msg, None, None]:
        """
        Drives sample_x and alignment_y to the edge of the loop. The edge of the loop is found
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
        x_coord, y_coord = self.find_loop_edge_coordinates()

        loop_position_sample_x = (
            self.sample_x.position
            - np.sin(np.radians(self.motor_phi.position)) * (x_coord - self.beam_position[0]) / self.zoom.pixels_per_mm
        )

        loop_position_sample_y = (
            self.sample_y.position
            - np.cos(np.radians(self.motor_phi.position)) * (x_coord - self.beam_position[0]) / self.zoom.pixels_per_mm
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
            loop_position_alignment_y)

    def find_loop_edge_coordinates(self) -> tuple[float, float]:
        """
        Finds the edge of the loop using either lucid3 or the loop
        finder code developed by PSI

        Returns
        -------
        tuple[float, float]
            The x and y pixel coordinates of the edge of the loop

        Raises
        ------
        An error if method is not lucid3 or psi
            
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
            procImg.findContour(
                zoom=self.loop_img_processing_zoom,
                beamline=self.loop_img_processing_beamline,
            )
            extremes = procImg.findExtremes()
            screen_coordinates = extremes["top"]
            x_coord = screen_coordinates[0]
            y_coord = screen_coordinates[1]
        else:
            raise NotImplementedError(
                f"Supported methods are lucid3 and psi, not {self.method}"
            )

        if self.plot:
            self.save_image(
                data,
                x_coord,
                y_coord,
                f"step_2_loop_centering_fig_{x_coord}",
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

        data = self.get_image_from_camera(np.uint8)

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
                "step_2_centered_loop",
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
        data = self.get_image_from_camera()

        try:
            gray_image = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
        except cv2.error:
            # The MD3 camera already returns black and white images for the zoom levels 5, 6 and 7,
            # so we don't do anything here
            gray_image = data

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
            logger.info(
                "WARNING! Camera connection timed out, sending a static image of the test rig"
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
        plt.scatter(x_coord, y_coord, s=200, c="r", marker="+", label=f"Omega={self.motor_phi.position}")
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
