from mx3_beamline_library.plans.calibration.trays.calibrate import calibrate_plate, move_to_well_spot
from mx3_beamline_library.plans.calibration.trays.plane_frame import PlaneFrame
from bluesky import RunEngine
from typing import Literal
RE = RunEngine()

def calibrate_and_move_move_to_well(plate_type: Literal["swissci_lowprofile", "mitegen_insitu", "swissci_highprofile", "mrc"], well: str):
    cal_plane = yield from calibrate_plate(plate_type)
    print("u:", cal_plane.u_axis)
    print("v:", cal_plane.v_axis)
    print("origin:", cal_plane.origin)

    yield from move_to_well_spot(well,cal_plane, plate_type)

# def move_to_well(u_axis, v_axis, origin, well: str, plate_type: Literal["swissci_lowprofile", "mitegen_insitu", "swissci_highprofile", "mrc"]):
#     # This function is a placeholder for the actual implementation
#     print(f"Moving to well {well} on plate type {plate_type} with u-axis {u_axis}, v-axis {v_axis}, and origin {origin}")

#     plane_frame = PlaneFrame(origin, u_axis, v_axis)

#     yield from move_to_well_spot(well, plane_frame, plate_type)


RE(calibrate_and_move_move_to_well(well="D6 well 1", plate_type="swissci_highprofile"))
