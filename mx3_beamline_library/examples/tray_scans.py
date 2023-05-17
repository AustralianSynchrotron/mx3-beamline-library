from os import environ

import requests
from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from mx3_beamline_library.plans.tray_scans import single_drop_grid_scan, multiple_drop_grid_scans
from mx3_beamline_library.devices.detectors import dectris_detector
from mx3_beamline_library.schemas.detector import UserData

environ["BL_ACTIVE"] = "True"
environ["MD3_ADDRESS"] = "10.244.101.30"
environ["MD3_PORT"] = "9001"
environ["MD_REDIS_HOST"] = "10.244.101.30"
environ["MD_REDIS_PORT"] = "6379"
environ["DECTRIS_DETECTOR_HOST"] = "10.244.101.200"
environ["DECTRIS_DETECTOR_PORT"] = "80"

# Instantiate run engine an start plan
RE = RunEngine({})
bec = BestEffortCallback()
RE.subscribe(bec)


user_data = UserData(tray_id="my_tray", zmq_consumer_mode="spotfinder")

#RE(single_drop_grid_scan(
#    detector=dectris_detector, column=0, row=0, drop=0, grid_number_of_columns=5, grid_number_of_rows=5,
#             exposure_time=1, user_data=user_data, alignment_z_offset=-1.0))

drop_locations = ["A1-1", "B1-1"]

RE(multiple_drop_grid_scans(
    detector=dectris_detector, drop_locations=drop_locations, grid_number_of_columns=5, grid_number_of_rows=5,
             exposure_time=1, user_data=user_data,alignment_y_offset=0.2, alignment_z_offset=-1.0))