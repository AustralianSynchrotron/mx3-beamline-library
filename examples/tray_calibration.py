from mx3_beamline_library.plans.calibration.trays.calibrate import calibrate_plate, move_to_well_spot
from bluesky import RunEngine
from typing import Literal
RE = RunEngine()

def calibrate_and_move_move_to_well(plate_type: Literal["swissci_lowprofile", "mitegen_insitu", "swissci_highprofile", "mrc"], well: str):
    cal_plane = yield from calibrate_plate(plate_type)

    yield from move_to_well_spot(well,cal_plane, plate_type)


RE(calibrate_and_move_move_to_well(well="A1 well 1", plate_type="swissci_lowprofile"))