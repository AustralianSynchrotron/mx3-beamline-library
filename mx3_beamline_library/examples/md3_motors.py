"""
This example gets the positions of all MD3 Motors,
and optionally runs a grid scan plan using MD3 motors.
"""

from os import environ

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.plans import grid_scan
from ophyd.sim import det1

environ["BL_ACTIVE"] = "True"
environ["MD3_ADDRESS"] = "10.244.101.30"
environ["MD3_PORT"] = "9001"
from mx3_beamline_library.devices.motors import md3  # noqa

# Change the following statement to True if you want to run a grid scan.
RUN_GRID_SCAN = False

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

# Optionally run a grid scan
if RUN_GRID_SCAN:
    RE = RunEngine()
    bec = BestEffortCallback()
    RE.subscribe(bec)

    RE(
        grid_scan(
            [det1],
            md3.sample_y,
            -0.1,
            0.1,
            2,
            md3.sample_x,
            -0.1,
            0.1,
            2,
            snake_axes=False,
            md={"sample_id": "test"},
        )
    )
