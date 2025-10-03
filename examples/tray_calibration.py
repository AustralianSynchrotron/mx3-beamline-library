from bluesky import RunEngine

from mx3_beamline_library.plans.calibration.trays.calibrate import (
    calibrate_plate,
    move_to_well_spot,
)

RE = RunEngine()


RE(calibrate_plate(plate_type="swissci_lowprofile"))

RE(move_to_well_spot(well_input="A11-2", plate_type="swissci_lowprofile"))
