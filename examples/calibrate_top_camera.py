from bluesky import RunEngine

from mx3_beamline_library.plans.calibration import (
    top_camera_pixels_per_mm,
    top_camera_target_coords,
)

top_camera_coords = top_camera_target_coords.TopCameraTargetCoords()

RE = RunEngine()

RE(top_camera_pixels_per_mm.set_x_and_y_pixels_per_mm())

RE(top_camera_coords.set_top_camera_target_coords())
