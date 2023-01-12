import typing
import pytest
from types import GeneratorType
from ophyd import Device
from bluesky.utils import Msg
from mx3_beamline_library.plans.stubs import stage_devices, unstage_devices

if typing.TYPE_CHECKING:
    from _pytest.fixtures import SubRequest


@pytest.fixture(scope="class")
def stub_devices(request: "SubRequest") -> list[Device]:
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

    devices: list[Device] = []
    for i in range(request.param):
        devices.append(Device(name=f"Device_{i}"))

    return devices


@pytest.mark.parametrize("stub_devices", [5], indirect=True)
class TestStubs:
    """Run stub tests"""

    def test_devices_stage(self, stub_devices: list[Device]):
        """_summary_

        Parameters
        ----------
        camera : BlackFlyCam
            _description_
        """

        res = stage_devices(devices=stub_devices)
        assert isinstance(res, GeneratorType)

        msgs = list(res)
        assert len(msgs) == len(stub_devices)

        for msg in msgs:
            assert isinstance(msg, Msg)

    def test_devices_unstage(self, stub_devices: list[Device]):
        """_summary_

        Parameters
        ----------
        camera : BlackFlyCam
            _description_
        """

        res = unstage_devices(devices=stub_devices)
        assert isinstance(res, GeneratorType)

        msgs = list(res)
        assert len(msgs) == len(stub_devices)

        for msg in msgs:
            assert isinstance(msg, Msg)
