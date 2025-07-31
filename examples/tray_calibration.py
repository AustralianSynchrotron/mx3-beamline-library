from mx3_beamline_library.plans.calibration.trays.calibrate import calibrate_plate
from bluesky import RunEngine

RE = RunEngine()
RE(calibrate_plate(plate_type="mitegen_insitu"))