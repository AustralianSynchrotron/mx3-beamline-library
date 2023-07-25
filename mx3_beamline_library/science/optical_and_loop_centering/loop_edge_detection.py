import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from ...schemas.loop_edge_detection import LoopExtremes, RectangleCoordinates


class LoopEdgeDetection:
    """
    This class is used to identify the edge of a loop and is based on code developed
    by PSI. To identify the edge of the loop, we filter the image using adaptive threshold,
    and then we find the biggest contour of the image.
    """

    def __init__(
        self, image: npt.NDArray, block_size: int, adaptive_constant: float
    ) -> None:
        """
        Parameters
        ----------
        image : npt.NDArray
            A numpy array
        block_size : int
            Size of a pixel neighborhood that is used to calculate a threshold
            value for the pixel: 3, 5, 7, and so on.
        adaptive_constant : float
            Constant subtracted from the mean or weighted mean.
            Normally, it is positive but may be zero or negative as well.

        Returns
        -------
        None
        """
        if image.dtype == np.uint8:
            self.image = image
        else:
            self.image = image.astype(np.uint8)

        self.block_size = block_size
        self.adaptive_constant = adaptive_constant
        self.contour = self._find_biggest_contour()

    def _convert_image_to_grayscale(self) -> npt.NDArray:
        """
        If self.image is an RGB image, we convert it to grayscale

        Returns
        -------
        npt.NDArray
            A grayscale image
        """
        if len(self.image.shape) == 2:
            image_gray = self.image
        else:
            image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return image_gray

    def _apply_threshold(self, gray_image: npt.NDArray) -> npt.NDArray:
        """
        Applies adaptive threshold to the image

        Parameters
        ----------
        gray_image : npt.NDArray
            A grayscale image

        Returns
        -------
        npt.NDArray
            The image after applying adaptive threshold
        """

        threshold = cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.block_size,
            self.adaptive_constant,
        )
        return threshold

    def _find_biggest_contour(self) -> npt.NDArray:
        """
        Finds the biggest contour of the image

        Returns
        -------
        npt.NDArray
            The biggest contour
        """
        gray_image = self._convert_image_to_grayscale()
        threshold = self._apply_threshold(gray_image)
        contours, hierarchy = cv2.findContours(
            image=threshold, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE
        )

        biggest_contour = max(contours, key=cv2.contourArea)
        return biggest_contour

    def find_tip(self) -> npt.NDArray:
        """
        Finds the (x,y) pixel coordinates of tip of the loop. We assume that the tip of
        the loop is on the top of the image

        Returns
        -------
        npt.NDArray
            The tip of the loop pixel coordinates
        """
        return self.contour[self.contour[:, :, 1].argmin()][0]

    def find_extremes(self) -> LoopExtremes:
        """
        Finds the (x,y) pixels coordinates of the extremes of the lop,
        namely: top, bottom, right and left

        Returns
        -------
        dict[str, npt.NDArray]
            A dictionary containing the tip of the loop
        """
        loop_extremes = LoopExtremes(
            top=self.contour[self.contour[:, :, 1].argmin()][0],
            bottom=self.contour[self.contour[:, :, 1].argmax()][0],
            right=self.contour[self.contour[:, :, 0].argmax()][0],
            left=self.contour[self.contour[:, :, 0].argmin()][0],
        )
        return loop_extremes

    def fit_rectangle(self) -> RectangleCoordinates:
        """
        Finds the top_left and bottom right coordinates of a rectangle based on the
        find_extremes method

        Returns
        -------
        dict[str, npt.NDArray]
            A dictionary containing the (x,y) top_left and bottom right coordinates of the
            rectangle surrounding the loop
        """
        extremes = self.find_extremes()

        rectangle_coordinates = RectangleCoordinates(
            top_left=np.array([extremes.left[0], extremes.top[1]]),
            bottom_right=np.array([extremes.right[0], extremes.bottom[1]]),
        )

        return rectangle_coordinates

    def loop_area(self) -> float:
        """
        Calculates the area of the loop

        Returns
        -------
        float
            The area of the loop
        """
        return cv2.contourArea(self.contour)

    def plot_raster_grid(
        self,
        rectangle_coordinates: RectangleCoordinates,
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
        data = self.image
        plt.imshow(data)

        # Plot grid:
        # Top
        plt.scatter(
            rectangle_coordinates.top_left[0],
            rectangle_coordinates.top_left[1],
            s=200,
            c="b",
            marker="+",
        )
        plt.scatter(
            rectangle_coordinates.bottom_right[0],
            rectangle_coordinates.bottom_right[1],
            s=200,
            c="b",
            marker="+",
        )

        # top
        x = np.linspace(
            rectangle_coordinates.top_left[0],
            rectangle_coordinates.bottom_right[0],
            100,
        )
        z = rectangle_coordinates.top_left[1] * np.ones(len(x))
        plt.plot(x, z, color="red", linestyle="--")

        # Bottom
        x = np.linspace(
            rectangle_coordinates.top_left[0],
            rectangle_coordinates.bottom_right[0],
            100,
        )
        z = rectangle_coordinates.bottom_right[1] * np.ones(len(x))
        plt.plot(x, z, color="red", linestyle="--")

        # Right side
        z = np.linspace(
            rectangle_coordinates.top_left[1],
            rectangle_coordinates.bottom_right[1],
            100,
        )
        x = rectangle_coordinates.bottom_right[0] * np.ones(len(x))
        plt.plot(x, z, color="red", linestyle="--")

        # Left side
        z = np.linspace(
            rectangle_coordinates.top_left[1],
            rectangle_coordinates.bottom_right[1],
            100,
        )
        x = rectangle_coordinates.top_left[0] * np.ones(len(x))
        plt.plot(x, z, color="red", linestyle="--")
        plt.savefig(filename)
        plt.close()
