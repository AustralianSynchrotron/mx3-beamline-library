"""
This example shows how to run grid scans on four different drop locations,
namely ["A1-1", "A2-1", "B1-1", "B2-1"]

    Requirements:
    - A Redis connection
    - Access to the Dectris SIMPLON API (or Simulated SIMPLON-API)

    Optional requirements:
    - Access to the MD3 exporter server. If the environment variable
    BL_ACTIVE=False, access to the server is not needed and ophyd
    simulated motors as used as a replacement.
"""


from os import environ

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback

# Modify the following ENV variables with the corresponding
# hosts and ports.
# IF BL_ACTIVE=False, we run the library in simulation mode
environ["BL_ACTIVE"] = "False"
environ["MD3_ADDRESS"] = "12.345.678.90"
environ["MD3_PORT"] = "1234"
environ["MD3_REDIS_HOST"] = "12.345.678.90"
environ["MD3_REDIS_PORT"] = "1234"
environ["SIMPLON_API"] = "http://0.0.0.0:8000"

from mx3_beamline_library.devices.detectors import dectris_detector  # noqa
from mx3_beamline_library.plans.tray_scans import multiple_drop_grid_scan  # noqa

# Instantiate run engine an start plan
RE = RunEngine({})
bec = BestEffortCallback()
RE.subscribe(bec)


RE(
    multiple_drop_grid_scan(
        tray_id="my_tray",
        detector=dectris_detector,
        drop_locations=["A1-1", "A2-1", "B1-1", "B2-1"],
        grid_number_of_columns=5,
        grid_number_of_rows=5,
        exposure_time=0.6,
        omega_range=0,
        alignment_y_offset=0.2,
        alignment_z_offset=-1.0,
    )
)
