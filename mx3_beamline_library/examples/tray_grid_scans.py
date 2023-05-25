"""
This example shows how to run grid scans on four different drop locations,
namely ["A1-1", "A2-1", "B1-1", "B2-1"]
"""

from os import environ

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback

from mx3_beamline_library.devices.detectors import dectris_detector
from mx3_beamline_library.plans.tray_scans import multiple_drop_grid_scan
from mx3_beamline_library.schemas.detector import UserData

environ["BL_ACTIVE"] = "True"
environ["MD3_ADDRESS"] = "12.345.678.90"
environ["MD3_PORT"] = "1234"
environ["MD_REDIS_HOST"] = "12.345.678.90"
environ["MD_REDIS_PORT"] = "1234"
environ["DECTRIS_DETECTOR_HOST"] = "127.0.0.1"
environ["DECTRIS_DETECTOR_PORT"] = "8000"

# Instantiate run engine an start plan
RE = RunEngine({})
bec = BestEffortCallback()
RE.subscribe(bec)


RE(
    multiple_drop_grid_scan(
        detector=dectris_detector,
        drop_locations=["A1-1", "A2-1", "B1-1", "B2-1"],
        grid_number_of_columns=5,
        grid_number_of_rows=5,
        exposure_time=0.6,
        omega_range=0,
        user_data=UserData(id="my_tray", zmq_consumer_mode="spotfinder"),
        alignment_y_offset=0.2,
        alignment_z_offset=-1.0,
    )
)
