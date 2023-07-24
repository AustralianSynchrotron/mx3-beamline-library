import cv2
import numpy as np
import numpy.typing as npt

class LoopEdgeDetection:
    """
    This class is used to identify the edge of a loop and it is based on code developed
    by PSI. To identify the edge of the loop, we apply an adaptive threshold to the image,
    which is then used to find the biggest contour of the image.
    """
    def __init__(self, image: npt.NDArray, block_size: int, adaptive_constant: float) -> None:
        """
        Parameters
        ----------
        image : npt.NDArray
            A numpy array of type np.uint8
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
        self.image = image
        self.block_size =block_size
        self.adaptive_constant = adaptive_constant
        self.contour = self._find_biggest_contour()

    def _convert_image_to_grayscale(self) -> npt.NDArray:
        """
        If self.image is and RGB image, we convert it to grayscale

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
    
    def _apply_threshold(self, gray_image) -> npt.NDArray:
        """
        Applies adaptive threshold to the image

        Parameters
        ----------
        gray_image : _type_
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
        
        biggest_contour = max(contours, key = cv2.contourArea)
        return biggest_contour
    
    def find_tip(self) -> npt.NDArray:
        """
        Finds the tip of the loop. We assume that the tip of the loop is always on the top
        of the image

        Returns
        -------
        npt.NDArray
            The tip of the loop pixel coordinates
        """
        return self.contour[self.contour[:, :, 1].argmin()][0]
    
    def find_extremes(self) -> dict[str, npt.NDArray]:
        """
        Finds the extremes of the lop, namely: top, bottom, right and left

        Returns
        -------
        dict[str, npt.NDArray]
            A dictionary containing the tip of the loop
        """
        leftmost = self.contour[self.contour[:, :, 0].argmin()][0]
        rightmost = self.contour[self.contour[:, :, 0].argmax()][0]
        topmost = self.contour[self.contour[:, :, 1].argmin()][0]
        bottommost = self.contour[self.contour[:, :, 1].argmax()][0]
        return {
            "top": topmost,
            "bottom": bottommost,
            "right": rightmost,
            "left": leftmost,
        }
    
    def fit_rectangle(self) -> dict[str, npt.NDArray]:
        """
        Based on three extreme points of loop contour (top, bottom, leftmost)
        finds rectangle bounding the loop.

        - the height of the box is the distance between top-most and bottom-most-point
        - the width of the the  box is the distance between left-most point and
          top-most or bottom-most point, which ever is bigger, multiplied by two.

        Returns
        -------
        dict[str, npt.NDArray]
            => keys: [top_left, bottom_left, top_right, bottom_right].
            => values numpy arrays with coordinates np.array[x,y]

        """
        extremes = self.find_extremes()

        rectangle = {}
        img_height, img_width = self.image.shape[:2]

        rectangle["top_left"] = np.array([extremes["left"][0], extremes["top"][1]])

        # Create rectangle based on extreme point that more away from the tip
        x_top_right_t = extremes["left"][0] + 2 * (
            extremes["top"][0] - extremes["left"][0]
        )
        x_top_right_b = extremes["left"][0] + 2 * (
            extremes["bottom"][0] - extremes["left"][0]
        )

        x_top_right = max(x_top_right_t, x_top_right_b)

        # If calculations are more than image width, use image width
        x_top_right = min(x_top_right, img_width)

        rectangle["bottom_right"] = np.array([x_top_right, extremes["bottom"][1]])

        return rectangle