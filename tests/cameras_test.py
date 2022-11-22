import typing
import pytest

if typing.TYPE_CHECKING:
    from _pytest.fixtures import SubRequest
    from ophyd import Device
    from mx3_beamline_library.devices.sim.classes.detectors import BlackFlyCam
    from mx3_beamline_library.devices.sim import detectors
    Detectors = detectors


@pytest.fixture(scope="class")
def camera(request: "SubRequest", detectors: "Detectors") -> "Device":
    """_summary_

    Parameters
    ----------
    request : SubRequest
        _description_
    cameras : Detectors
        _description_

    Returns
    -------
    MX3SimMotor
        _description_
    """

    camera_name = request.param
    camera: "Device" = getattr(detectors, camera_name)
    camera.wait_for_connection(timeout=5)

    return camera


# @pytest.mark.parametrize("camera", ["blackfly_camera"], indirect=True)
# class TestBlackFlyCam:
#     """Run BlackFlyCam tests."""

#     def test_camera_setup(self, camera: "BlackFlyCam"):
#         """_summary_

#         Parameters
#         ----------
#         camera : BlackFlyCam
#             _description_
#         """

#         assert camera is not None
