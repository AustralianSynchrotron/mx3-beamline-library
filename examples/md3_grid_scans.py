"""
This example shows bluesky plans that trigger MD3 grid scans and 4D scans.
    Requirements:
        - A Redis connection
        - Access to the Dectris SIMPLON API
        - Access to the MD3 exporter server
"""

from os import environ

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback

# Modify the following ENV variables with the corresponding
# hosts and ports.
environ["BL_ACTIVE"] = "True"
environ["MD3_ADDRESS"] = "12.345.678.90"
environ["MD3_PORT"] = "9001"
environ["MD3_REDIS_HOST"] = "12.345.678.90"
environ["MD3_REDIS_PORT"] = "6379"
environ["DECTRIS_DETECTOR_HOST"] = "12.345.678.90"
environ["DECTRIS_DETECTOR_PORT"] = "8000"

from mx3_beamline_library.devices.detectors import dectris_detector  # noqa
from mx3_beamline_library.plans.basic_scans import md3_4d_scan, md3_grid_scan  # noqa
from mx3_beamline_library.schemas.detector import UserData  # noqa

RE = RunEngine()
bec = BestEffortCallback()
RE.subscribe(bec)

scan_type = "md3_4d_scan"

if scan_type == "md3_grid_scan":
    user_data = UserData(
        id="my_sample",
        zmq_consumer_mode="spotfinder",
        grid_scan_id="flat",
    )
    RE(
        md3_grid_scan(
            detector=dectris_detector,
            grid_width=0.7839332119645885,
            grid_height=0.49956528213429663,
            number_of_columns=4,
            number_of_rows=4,
            start_omega=176.7912087912088,
            omega_range=0,
            start_alignment_y=-0.010115684286529321,
            start_alignment_z=0.6867517681659011,
            start_sample_x=-0.10618655152995649,
            start_sample_y=-0.4368335669982139,
            md3_exposure_time=1,
            user_data=user_data,
        )
    )

elif scan_type == "md3_4d_scan":
    user_data = UserData(
        id="my_sample", zmq_consumer_mode="spotfinder", grid_scan_id="edge"
    )
    RE(
        md3_4d_scan(
            detector=dectris_detector,
            start_angle=176.7912087912088,
            scan_range=0,
            md3_exposure_time=2,
            start_alignment_y=-0.010115684286529321,
            stop_alignment_y=0.57,
            start_sample_x=-0.10618655152995649,
            stop_sample_x=-0.10618655152995649,
            start_sample_y=-0.106,
            stop_sample_y=-0.106,
            start_alignment_z=1.1,
            stop_alignment_z=1.1,
            number_of_frames=8,
            user_data=user_data,
        )
    )
