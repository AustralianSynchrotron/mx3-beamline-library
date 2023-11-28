"""
This example gets the positions of all MD3 Motors.

    Requirements:
    - Access to the MD3 exporter server
"""

from os import environ

# Modify the following ENV variables with the corresponding
# hosts and ports.
environ["BL_ACTIVE"] = "True"
environ["MD3_ADDRESS"] = "12.345.678.90"
environ["MD3_PORT"] = "9001"
environ["MD3_REDIS_HOST"] = "12.345.678.90"
environ["MD3_REDIS_PORT"] = "6379"
from mx3_beamline_library.devices.detectors import md_camera  # noqa
from mx3_beamline_library.devices.motors import md3  # noqa

# Print motor positions
print(f"sample x: {md3.sample_x.position}")
print(f"sample y: {md3.sample_y.position}")
print(f"alignment x: {md3.alignment_x.position}")
print(f"alignment y: {md3.alignment_y.position}")
print(f"alignment z: {md3.alignment_z.position}")
print(f"omega: {md3.omega.position}")
print(f"kappa: {md3.kappa.position}")
print(f"phi (a.k.a kappa phi): {md3.phi.position}")
print(f"aperture vertical: {md3.aperture_vertical.position}")
print(f"aperture horizontal: {md3.aperture_horizontal.position}")
print(f"capillary vertical: {md3.capillary_vertical.position}")
print(f"capillary horizontal: {md3.capillary_horizontal.position}")
print(f"scintillator vertical: {md3.scintillator_vertical.position}")
print(f"scintillator vertical: {md3.scintillator_vertical.position}")
print(f"zoom: {md3.zoom.position}, pixels_per_mm: {md3.zoom.pixels_per_mm}")

# Print camera width and height
print(f"Camera width and height: {(md_camera.width, md_camera.height)}")
