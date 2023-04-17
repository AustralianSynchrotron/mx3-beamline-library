import numpy as np
import numpy.typing as npt
from os import path


""" Library scoped constants. """
PV_PREFIX = "SR##ID"

TOP_CAMERA_BACKGROUND_IMAGE = path_to_config_file = path.join(
    path.dirname(__file__), "plans/configuration/top_camera_background_image.npy"
)

top_camera_background_img_array: npt.NDArray = np.load(
    TOP_CAMERA_BACKGROUND_IMAGE 
)
