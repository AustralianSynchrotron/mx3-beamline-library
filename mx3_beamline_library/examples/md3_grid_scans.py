"""
This example shows how to run three different grid scans
"""
from os import environ

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback

# Modify the following ENV variables with the corresponding
# hosts and ports
environ["BL_ACTIVE"] = "True"
environ["MD3_ADDRESS"] = "12.345.678.90"
environ["MD3_PORT"] = "9001"
environ["MD_REDIS_HOST"] = "12.345.678.90"
environ["MD_REDIS_PORT"] = "6379"
environ["DECTRIS_DETECTOR_HOST"] = "12.345.678.90"
environ["DECTRIS_DETECTOR_PORT"] = "8000"

from mx3_beamline_library.devices.detectors import dectris_detector  # noqa
from mx3_beamline_library.devices.motors import md3  # noqa
from mx3_beamline_library.plans.basic_scans import (  # noqa
    md3_4d_scan,
    md3_grid_scan,
    test_md3_grid_scan_plan,
)
from mx3_beamline_library.schemas.detector import UserData  # noqa
from mx3_beamline_library.schemas.xray_centering import RasterGridCoordinates  # noqa

RE = RunEngine()
bec = BestEffortCallback()
RE.subscribe(bec)

scan_type = "md3_4d_scan"

if scan_type == "md3_grid_scan":
    user_data = UserData(
        sample_id="my_sample", zmq_consumer_mode="spotfinder", grid_scan_type="flat"
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
            exposure_time=1,
            user_data=user_data,
        )
    )

elif scan_type == "md3_4d_scan":
    user_data = UserData(
        sample_id="my_sample", zmq_consumer_mode="spotfinder", grid_scan_type="edge"
    )
    RE(
        md3_4d_scan(
            detector=dectris_detector,
            start_angle=176.7912087912088,
            scan_range=0,
            exposure_time=2,
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

elif scan_type == "test_md3_grid_scan":
    raster_grid_coords = RasterGridCoordinates(
        initial_pos_sample_x=-0.022731250443299555,
        final_pos_sample_x=-0.10983893569861315,
        initial_pos_sample_y=0.6242099418914737,
        final_pos_sample_y=0.7824280466265174,
        initial_pos_alignment_y=0.009903480128227239,
        final_pos_alignment_y=0.43069116007980784,
        center_pos_sample_x=-0.06628509307095636,
        center_pos_sample_y=0.7033189942589956,
        width=0.1806120635408611,
        height=0.4207876799515806,
        number_of_columns=3,
        number_of_rows=3,
        omega=0,
    )

    RE(
        test_md3_grid_scan_plan(
            raster_grid_coords, md3.alignment_y, md3.sample_x, md3.sample_y, md3.omega
        )
    )
