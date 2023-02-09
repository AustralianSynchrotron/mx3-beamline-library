import logging
from typing import Generator, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from bluesky.plan_stubs import mv
from bluesky.utils import Msg
from ophyd.signal import ConnectionTimeoutError
from scipy import optimize

from mx3_beamline_library.devices.classes.detectors import BlackFlyCam, MDRedisCam
from mx3_beamline_library.devices.classes.motors import (
    CosylabMotor,
    MD3BackLight,
    MD3Motor,
    MD3Phase,
    MD3Zoom,
)
from mx3_beamline_library.science.optical_and_loop_centering.psi_optical_centering import (
    loopImageProcessing,
)

from ..schemas.optical_and_xray_centering import CenteredLoopMotorCoordinates

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
        camera: Union[BlackFlyCam, MDRedisCam],
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
        auto_focus: bool = True,
        min_focus: float = -0.3,
        max_focus: float = 1.3,
        tol: float = 0.3,
        number_of_intervals: int = 2,
        plot: bool = False,
        loop_img_processing_beamline: str = "MX3",
        loop_img_processing_zoom: str = "1",
        number_of_omega_steps: int = 7,
    ) -> None:
        """
        Parameters
        ----------
        camera : Union[BlackFlyCam, MDRedisCam]
            Camera
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
        self.omega = omega
        self.zoom = zoom
        self.phase = phase
        self.backlight = backlight
        self.beam_position = beam_position
        self.auto_focus = auto_focus
        self.min_focus = min_focus
        self.max_focus = max_focus
        self.tol = tol
        self.number_of_intervals = number_of_intervals
        self.plot = plot
        self.loop_img_processing_beamline = loop_img_processing_beamline
        self.loop_img_processing_zoom = loop_img_processing_zoom
        self.number_of_omega_steps = number_of_omega_steps

        self.centered_loop_position = None

    def center_loop(self):
        """
        This plan is the main optical loop centering plan, which is used by the
        optical_and_xray_centering plan. Here, we optically center the loop using the loop
        centering code developed by PSI. Before analysing an image, we unblur the
        image at to make sure the results are consistent.

        Yields
        ------
        Generator[Msg, None, None]
            A plan that automatically centers a loop
        """
        # Set phase to `Centring`
        current_phase = self.phase.get()
        if current_phase != "Centring":
            yield from mv(self.phase, "Centring")

        # Drive the motors, zoom and backlight to the default start positions
        yield from mv(
            self.alignment_x,
            0.434,
            self.alignment_y,
            0.1,
            self.alignment_z,
            0.63,
            self.sample_x,
            0.2,
            self.sample_y,
            0.35,
            self.omega,
            0,
            self.zoom,
            1,
            self.backlight,
            2,
        )

        x_coords, y_coords, omega_positions = [], [], []

        # The zoom list allows us to add more precision at higher zoom levels
        # if we need to in the future, e.g, zoom_list = [1, 4],
        #  at the moment [1] works well
        zoom_list = [1]
        for zoom_value in zoom_list:
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

                x_coords.append(x / self.zoom.pixels_per_mm)
                y_coords.append(y / self.zoom.pixels_per_mm)
                omega_positions.append(np.radians(self.omega.position))

            yield from self.drive_motors_to_aligned_position(
                x_coords, y_coords, omega_positions
            )
            self.centered_loop_position = CenteredLoopMotorCoordinates(
                alignment_x=self.alignment_x.position,
                alignment_y=self.alignment_y.position,
                alignment_z=self.alignment_z.position,
                sample_x=self.sample_x.position,
                sample_y=self.sample_y.position,
            )

        yield from self.find_edge_and_flat_angles()

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

        r, a, offset = self.multiPointCentre(np.array(z).flatten(), omega_positions)
        dy = r * np.sin(a)
        dx = r * np.cos(a)

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

    def multiPointCentre(self, z: npt.NDArray, omega_list: list):
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
            The solution to the error function `errfunc`
        """

        def fitfunc(p, x):
            return p[0] * np.sin(x + p[1]) + p[2]

        def errfunc(p, x, y):
            return fitfunc(p, x) - y

        # The function call returns tuples of varying length
        result = optimize.leastsq(errfunc, [1.0, 0.0, 0.0], args=(omega_list, z))
        return result[0]

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
        data = self.get_image_from_camera(np.uint8)

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
            self.save_image(
                data,
                x_coord,
                y_coord,
                f"step_2_loop_centering_{round(self.omega.position)}",
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
            # The MD3 camera already returns black and white images for the zoom levels
            # 5, 6 and 7, so we don't do anything here
            gray_image = data

        return cv2.Laplacian(gray_image, cv2.CV_64F).var()

    def get_image_from_camera(
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
            array_data: npt.NDArray = self.camera.array_data.get()
            if reshape:
                data = array_data.reshape(
                    self.camera.height.get(),
                    self.camera.width.get(),
                    self.camera.depth.get(),
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
        omega_list = np.linspace(0, 180, self.number_of_omega_steps)  # degrees
        area_list = []

        # We zoom in and increase the backlight intensity to improve accuracy
        yield from mv(self.zoom, 4, self.backlight, 2)

        for omega in omega_list:
            yield from mv(self.omega, omega)

            image = self.get_image_from_camera(np.uint8)
            procImg = loopImageProcessing(image)
            procImg.findContour(
                zoom=self.loop_img_processing_zoom,
                beamline=self.loop_img_processing_beamline,
            )
            extremes = procImg.findExtremes()
            area_list.append(self.quadrilateral_area(extremes))

        # Remove nans from list, and normalize the data (we do not care about amplitude,
        # we only care about phase)
        non_nan_args = np.invert(np.isnan(np.array(area_list)))
        omega_list = omega_list[non_nan_args]
        area_list = np.array(area_list)[non_nan_args]
        area_list = area_list / np.linalg.norm(area_list)

        # Fit the curve
        optimised_params, _ = optimize.curve_fit(
            self.sine_function,
            np.radians(omega_list),
            np.array(area_list),
            p0=[10, 0, 10],
            maxfev=4000,
        )

        x_new = np.linspace(0, 2 * np.pi, 4096)  # radians
        y_new = self.sine_function(
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
            plt.savefig("loop_area_curve_fit")
            plt.close()

    def sine_function(
        self, theta: float, amplitude: float, phase: float, offset: float
    ) -> float:
        """
        Sine function used to find the angles at which the area of a loop
        is maximum and minimum:

        area = amplitude*np.sin(omega*theta + phase) + offset

        Note that the period of the sine function is, by definition, T=pi, therefore
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
