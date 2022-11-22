import typing
import os.path
import pytest
from typing import Any, Dict
from ophyd.status import Status

if typing.TYPE_CHECKING:
    from requests import Response
    from _pytest.fixtures import SubRequest
    from mx3_beamline_library.devices.classes.detectors import DectrisDetector
    from mx3_beamline_library.devices.sim import detectors
    Detectors = detectors


@pytest.fixture(scope="class")
def detector(request: "SubRequest", detectors: "Detectors") -> "DectrisDetector":
    """Pytest fixture to load detector Ophyd device.

    Parameters
    ----------
    request : SubRequest
        Pytest subrequest parameters.
    detectors : Detectors
        Loaded detector module, either simulated or real.

    Returns
    -------
    DectrisDetector
        Dectris detector device instance.
    """

    # Load Ophyd device
    detector_name = request.param
    detector: "DectrisDetector" = getattr(detectors, detector_name)
    detector.wait_for_connection(timeout=5)
    yield detector


@pytest.mark.parametrize("detector", ["dectris_detector"], indirect=True)
class TestDectrisDetector:
    """Run DectrisDetector tests"""

    def test_detector_setup(self, detector: "DectrisDetector"):
        """Test detector device initialised.

        Parameters
        ----------
        detector : DectrisDetector
            Dectris detector device instance.
        """

        assert detector is not None

        resp: "Response" = detector.mock_get(f"{detector.REST}/")
        assert resp.status_code == 200

    @pytest.mark.parametrize(
        "config",
        [
            {},
            {"frame_time": 8},
            {"frame_time": 4, "nimages": 2},
            {"frame_time": "4", "nimages": 2.0},
        ],
    )
    def test_configure(self, detector: "DectrisDetector", config: Dict[str, Any]):
        """Test configuring detector device.

        Parameters
        ----------
        detector : DectrisDetector
            Dectris detector device instance.
        config : Dict[str, Any]
            _description_
        """

        old_config, new_config = detector.configure(detector_configuration=config)

        # Check return values
        assert isinstance(old_config, dict) and isinstance(new_config, dict)

        # Check context vars
        for key, value in config.items():
            resp: "Response" = detector.mock_get(os.path.join(detector.REST, "detector/api/1.8.0/config", key))
            assert resp.status_code == 200
            resp_dict: dict = resp.json()
            assert resp_dict.get("value") == int(value)

    def test_stage(self, detector: "DectrisDetector"):
        """Test arming the detector device.

        Parameters
        ----------
        detector : DectrisDetector
            Dectris detector device instance.
        """

        # Check to see if running mocked detector
        resp: "Response" = detector.mock_get(f"{detector.REST}/")
        assert resp.status_code == 200
        resp_dict: dict = resp.json() or {}
        api_type = resp_dict.get("SIMplonAPI")

        # Starting sequence_id value
        sequence_id_start: int
        if api_type == "Mocked":
            resp: "Response" = detector.mock_get(os.path.join(detector.REST, "metadata"))
            assert resp.status_code == 200
            resp_dict: dict = resp.json()
            sequence_id_start = resp_dict["metadata"]["sequence_id"]

        # Arm detector
        detector.stage()

        # Compare results
        if api_type == "Mocked":
            resp: "Response" = detector.mock_get(os.path.join(detector.REST, "metadata"))
            assert resp.status_code == 200
            resp_dict: dict = resp.json()
            assert resp_dict["metadata"]["sequence_id"] == sequence_id_start + 1
            assert resp_dict["metadata"]["image_number"] == 0

    def test_trigger(self, detector: "DectrisDetector"):
        """Test triggering the detector device.

        Parameters
        ----------
        detector : DectrisDetector
            Dectris detector device instance.
        """

        status = detector.trigger()

        assert isinstance(status, Status)

    def test_unstage(self, detector: "DectrisDetector"):
        """Test disarming the detector device.

        Parameters
        ----------
        detector : DectrisDetector
            Dectris detector device instance.
        """

        detector.unstage()
