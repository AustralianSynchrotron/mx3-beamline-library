from bluesky import RunEngine

from mx3_beamline_library.plans.calibration.trays.calibrate import move_to_well_spot, calibrate_plate

RE = RunEngine()


RE(calibrate_plate(plate_type="swissci_lowprofile"))

RE(move_to_well_spot(well_input="H1 well 1", plate_type="swissci_lowprofile"))
