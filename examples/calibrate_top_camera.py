from bluesky import RunEngine

from opentelemetry.trace import get_tracer

tracer = get_tracer(__name__)

from mx3_beamline_library.plans.calibration import (
    top_camera_pixels_per_mm,
    top_camera_target_coords,
)

top_camera_coords = top_camera_target_coords.TopCameraTargetCoords()

RE = RunEngine()

# individual telemetry for each plan
RE(top_camera_pixels_per_mm.set_x_and_y_pixels_per_mm())
RE(top_camera_coords.set_top_camera_target_coords())

# Or you can combine these together if you want combined telemetry
def full_top_camera_calibration():
    with tracer.start_as_current_span("full_top_camera_calibration"):
        yield from top_camera_pixels_per_mm.set_x_and_y_pixels_per_mm()
        yield from top_camera_coords.set_top_camera_target_coords()

RE(full_top_camera_calibration())