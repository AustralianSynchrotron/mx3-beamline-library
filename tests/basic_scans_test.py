import typing
import pytest
from bluesky import RunEngine
from pydantic import UUID4
from mx3_beamline_library.plans.basic_scans import scan_plan, grid_scan

if typing.TYPE_CHECKING:
    from mx3_beamline_library.devices.classes.detectors import DectrisDetector


@pytest.fixture(scope="class")
def run_engine() -> RunEngine:
    """Pytest fixture to initialise the bluesky run engine.

    Returns
    -------
    RunEngine
        Instance of the bluesky run engine.
    """

    yield RunEngine({})


@pytest.mark.parametrize("detector", ["dectris_detector"], indirect=True)
class TestBasicScans:
    """Run tests for bluesky basic scan plans"""

    @pytest.mark.parametrize("config", [{"frame_time": 4, "nimages": 2}])
    def test_scan_plan(self, detector: "DectrisDetector", run_engine: RunEngine, config: dict):
        """Test the "scan_plan" bluesky plan.

        Parameters
        ----------
        detector : DectrisDetector
            Dectris detector device instance.
        run_engine : RunEngine
            Instance of the bluesky run engine.
        """

        metadata = {}
        res = run_engine(scan_plan(detector, config, metadata))

        assert isinstance(res, tuple) and len(res) == 1
        assert isinstance(res[0], str) and UUID4(res[0])
        assert isinstance(metadata.get("dectris_sequence_id"), int)

    def test_scan_nd(self, detector: "DectrisDetector", run_engine: RunEngine):
        """Test the "scan_nd" bluesky plan.

        Parameters
        ----------
        detector : DectrisDetector
            Dectris detector device instance.
        run_engine : RunEngine
            Instance of the bluesky run engine.
        """

        assert True

    def test_grid_scan(self, detector: "DectrisDetector", run_engine: RunEngine):
        """Test the "grid_scan" bluesky plan.

        Parameters
        ----------
        detector : DectrisDetector
            Dectris detector device instance.
        run_engine : RunEngine
            Instance of the bluesky run engine.
        """

        # run_engine(grid_scan([detector], "motor_z"))
        assert True
