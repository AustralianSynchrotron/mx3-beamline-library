"""
This examples shows how to find the tip of a loop, and its corresponding
grid coordinates for raster scanning
"""

from mx3_beamline_library.devices.sim.classes.detectors import SIM_MD3_CAMERA_IMG
from mx3_beamline_library.science.optical_and_loop_centering.loop_edge_detection import (
    LoopEdgeDetection,
)

loop_edge_detections = LoopEdgeDetection(
    image=SIM_MD3_CAMERA_IMG, block_size=35, adaptive_constant=3
)

# Find the tip of a loop
tip = loop_edge_detections.find_tip()
print("tip of the loop", tip)

# Find the raster grid coordinates
rectangle_coordinates = loop_edge_detections.fit_rectangle()
print("Rectangle coordinates", rectangle_coordinates)

# Plot raster grid
loop_edge_detections.plot_raster_grid(rectangle_coordinates, filename="result")
