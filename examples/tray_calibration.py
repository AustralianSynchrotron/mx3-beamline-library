from mx3_beamline_library.plans.calibration.trays.calibrate import calibrate_plate, move_to_well_spot
from bluesky import RunEngine
from mx3_beamline_library.plans.calibration.trays.plate_configs.plate_configs import swissci_lowprofile

RE = RunEngine()

def calibrate_and_move_move_to_well(plate_type):
    cal_plane = yield from calibrate_plate(plate_type=plate_type)

    # Move to a specific well
    yield from move_to_well_spot("A1 well 1",cal_plane, swissci_lowprofile)


RE(calibrate_and_move_move_to_well(plate_type="swissci_lowprofile"))