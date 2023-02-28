"""
This example shows how to run three different grid scans
"""
from os import environ

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback

environ["BL_ACTIVE"] = "True"
environ["MD3_ADDRESS"] = "10.244.101.30"
environ["MD3_PORT"] = "9001"
environ["MD_REDIS_HOST"] = "10.244.101.30"
environ["MD_REDIS_PORT"] = "6379"
environ["DECTRIS_DETECTOR_HOST"] = "0.0.0.0"
environ["DECTRIS_DETECTOR_PORT"] = "8000"

from mx3_beamline_library.devices.detectors import dectris_detector  # noqa
from mx3_beamline_library.devices.motors import md3  # noqa
from mx3_beamline_library.plans.basic_scans import (  # noqa
    md3_4d_scan,
    md3_grid_scan,
    test_md3_grid_scan_plan,
)
from mx3_beamline_library.schemas.optical_and_xray_centering import (  # noqa
    RasterGridMotorCoordinates,
)

RE = RunEngine()
bec = BestEffortCallback()
RE.subscribe(bec)

scan_type = "test_md3_grid_scan"

if scan_type == "md3_grid_scan":
    RE(
        md3_grid_scan(
            detector=dectris_detector,
            detector_configuration={"nimages": 1},
            metadata={"sample_id": "sample_test"},
            grid_width=0.078777602182716,
            grid_height=0.4861154476152963,
            number_of_columns=2,
            number_of_rows=6,
            start_omega=30,
            start_alignment_y=0.224,
            start_alignment_z=0.69,
            start_sample_x=0.10100885629034344,
            start_sample_y=1.3073655240154325,
            exposure_time=1,
        )
    )

elif scan_type == "md3_4d_scan":
    RE(
        md3_4d_scan(
            detector=dectris_detector,
            detector_configuration={"nimages": 1},
            metadata={"sample_id": "sample_test"},
            start_angle=0,
            scan_range=0,
            exposure_time=3,
            start_alignment_y=0.29,
            stop_alignment_y=0.89,
            start_sample_x=-0.25,
            stop_sample_x=-0.25,
            start_sample_y=0.867,
            stop_sample_y=0.867,
            start_alignment_z=0.627,
            stop_alignment_z=0.627,
        )
    )

elif scan_type == "test_md3_grid_scan":
    raster_grid_coords = RasterGridMotorCoordinates(
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
