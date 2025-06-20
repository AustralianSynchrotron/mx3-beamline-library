from os import path

import numpy as np
import numpy.typing as npt

""" Library scoped constants. """

TOP_CAMERA_BACKGROUND_IMAGE = path.join(
    path.dirname(__file__), "plans/configuration/top_camera_background_image.npy"
)
top_camera_background_img_array: npt.NDArray = np.load(TOP_CAMERA_BACKGROUND_IMAGE)


# Detector
DETECTOR_PIXEL_SIZE_X = 7.5e-05  # meters
DETECTOR_PIXEL_SIZE_Y = 7.5e-05  # meters

DETECTOR_WIDTH_16M = 4148
DETECTOR_HEIGHT_16M = 4362

DETECTOR_WIDTH_4M = 2068
DETECTOR_HEIGHT_4M = 2162
