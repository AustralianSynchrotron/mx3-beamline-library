from os import path

import numpy as np
import numpy.typing as npt

""" Library scoped constants. """

TOP_CAMERA_BACKGROUND_IMAGE = path.join(
    path.dirname(__file__), "plans/configuration/top_camera_background_image.npy"
)
top_camera_background_img_array: npt.NDArray = np.load(TOP_CAMERA_BACKGROUND_IMAGE)
