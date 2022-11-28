import typing
import pytest
from typing import Generator, Optional, Callable
from collections import namedtuple
from cycler import Cycler
from bluesky import RunEngine
from bluesky.utils import Msg
from pydantic import UUID4
from bluesky.plans import plan_patterns
from mx3_beamline_library.plans.basic_scans import scan_plan, scan_nd, grid_scan

if typing.TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture
    from mx3_beamline_library.devices.sim import motors
    from mx3_beamline_library.devices.classes.detectors import DectrisDetector
    Motors = motors

MotorMoveAxis = namedtuple("MotorMoveAxis", ("axis", "initial", "final", "cells"))


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

    @staticmethod
    def _dummy_plan(*args: tuple, **kwargs: dict) -> Generator[Msg, None, None]:
        """Acts as a dummy bluesky plan.

        Yields
        ------
        Generator[Msg, None, None]
            Generator of bluesky message objects.
        """

        yield Msg(command="open_run")
        yield Msg(command="close_run", exit_status=None, reason=None)

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

    def test_scan_nd(self, detector: "DectrisDetector", run_engine: RunEngine, motors: "Motors"):
        """Test the "scan_nd" bluesky plan.

        Parameters
        ----------
        detector : DectrisDetector
            Dectris detector device instance.
        run_engine : RunEngine
            Instance of the bluesky run engine.
        motors : Motors
            Loaded motors module, either simulated or real.
        """

        testrig_z = MotorMoveAxis(motors.testrig.z, 0, 10, 5)
        testrig_x = MotorMoveAxis(motors.testrig.x, 0, 10, 5)
        args = [*testrig_z, *testrig_x, False]
        full_cycler = plan_patterns.outer_product(args=args)
        metadata = {
            "shape": (testrig_z.cells, testrig_x.cells),
            "extents": ([testrig_z.initial, testrig_z.final], [testrig_x.initial, testrig_x.final]),
            "snaking": (False, False),
            "plan_args": {
                "detectors": [detector],
                "args": args,
                "per_step": "None",
            },
            "plan_name": "grid_scan",
            "plan_pattern": "outer_product",
            "plan_pattern_args": {
                "args": args,
            },
            "plan_pattern_module": "bluesky.plan_patterns",
            "motors": (
                "testrig_z",
                "testrig_x",
            ),
            "hints": {
                "gridding": "rectilinear",
                "dimensions": [
                    (["testrig_z"], "primary"),
                    (["testrig_x"], "primary"),
                ],
            },
        }
        res = run_engine(scan_nd([detector], full_cycler, per_step=None, md=metadata))

        assert isinstance(res, tuple) and len(res) == 1
        assert isinstance(res[0], str) and UUID4(res[0])

    def test_grid_scan(self, detector: "DectrisDetector", run_engine: RunEngine, motors: "Motors", mocker: "MockerFixture"):
        """Test the "grid_scan" bluesky plan.

        Parameters
        ----------
        detector : DectrisDetector
            Dectris detector device instance.
        run_engine : RunEngine
            Instance of the bluesky run engine.
        motors : Motors
            Loaded motors module, either simulated or real.
        """

        def scan_nd_patched(detectors: list["DectrisDetector"], cycler: Cycler, *, per_step: Optional[Callable] = None, md: Optional[dict] = None) -> Generator[Msg, None, None]:
            """Patched "scan_nd" plan.

            Checks input parameters, then returns a dummy plan.

            Parameters
            ----------
            detectors : list[DectrisDetector]
                List of detector instances.
            cycler : Cycler
                _description_
            per_step : Optional[Callable], optional
                Optional callback object, by default None
            md : Optional[dict], optional
                Metadata dictionary, by default None

            Yields
            ------
            Generator[Msg, None, None]
                Generator of bluesky message objects.
            """

            # Sanity check parameters
            assert isinstance(detectors, list) and len(detectors) == 1
            assert detectors[0] == detector
            assert isinstance(cycler, Cycler)
            assert callable(per_step) or per_step is None
            assert isinstance(md, dict) or md is None

            # Yield dummy plan
            yield from self._dummy_plan()

        # Patch "scan_nd" with dummy plan, don't need to test it again
        mocker.patch("mx3_beamline_library.plans.basic_scans.scan_nd", side_effect=scan_nd_patched)

        metadata = {}
        res = run_engine(
            grid_scan(
                [detector],
                motors.testrig.z,
                0,
                10,
                5,
                motors.testrig.x,
                0,
                10,
                5,
                md=metadata,
            )
        )

        assert isinstance(res, tuple) and len(res) == 1
        assert isinstance(res[0], str) and UUID4(res[0])
