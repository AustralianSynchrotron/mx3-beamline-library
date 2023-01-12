import typing
import pytest
from typing import Generator, Optional, Callable, Union
from collections import namedtuple
from cycler import Cycler
from bluesky.utils import Msg
from pydantic import UUID4
from bluesky.plans import plan_patterns
from bluesky.plan_stubs import one_nd_step
from mx3_beamline_library.plans.basic_scans import scan_plan, scan_nd, grid_scan

if typing.TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture
    from bluesky import RunEngine
    from mx3_beamline_library.devices.sim import motors
    from mx3_beamline_library.devices.classes.detectors import DectrisDetector
    Motors = motors

MotorMoveAxis = namedtuple("MotorMoveAxis", ("axis", "initial", "final", "cells", "snaking"))


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
    def test_scan_plan(self, detector: "DectrisDetector", run_engine: "RunEngine", config: dict):
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

    @pytest.mark.parametrize(
        "testrig_z,testrig_x,per_step",
        (
            ((0, 10, 5, False), (0, 10, 5, False), None),
            ((0, 8, 4, True), (0, 8, 4, False), one_nd_step),
            ((2, 8, 2, True), (4, 8, 4, True), None),
        )
    )
    def test_scan_nd(
        self,
        detector: "DectrisDetector",
        run_engine: "RunEngine",
        motors: "Motors",
        testrig_z: tuple,
        testrig_x: tuple,
        per_step: Optional[Callable],
    ):
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

        testrig_z = MotorMoveAxis(motors.testrig.z, *testrig_z)
        testrig_x = MotorMoveAxis(motors.testrig.x, *testrig_x)
        args = [*testrig_z[:-1], *testrig_x[:-1], False]
        full_cycler = plan_patterns.outer_product(args=args)
        metadata = {
            "shape": (testrig_z.cells, testrig_x.cells),
            "extents": (
                [testrig_z.initial, testrig_z.final],
                [testrig_x.initial, testrig_x.final],
            ),
            "snaking": (testrig_z.snaking, testrig_x.snaking),
            "plan_args": {
                "detectors": [detector],
                "args": args,
                "per_step": str(per_step),
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
        res = run_engine(scan_nd([detector], full_cycler, per_step=per_step, md=metadata))

        assert isinstance(res, tuple) and len(res) == 1
        assert isinstance(res[0], str) and UUID4(res[0])

    @pytest.mark.parametrize("snake_axes,per_step", ((True, None), (False, None), ((("testrig", "x"),), one_nd_step)))
    def test_grid_scan(
        self,
        detector: "DectrisDetector",
        run_engine: "RunEngine",
        motors: "Motors",
        mocker: "MockerFixture",
        snake_axes: Union[bool, tuple[tuple[str]]],
        per_step: Optional[Callable],
    ):
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

        def scan_nd_patched(
            detectors: list["DectrisDetector"],
            cycler: Cycler,
            *,
            per_step: Optional[Callable] = None,
            md: Optional[dict] = None,
        ) -> Generator[Msg, None, None]:
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

        if isinstance(snake_axes, tuple) and isinstance(snake_axes[0], tuple):
            snake_axes = [
                getattr(
                    getattr(motors, device),
                    axis
                ) for device, axis in snake_axes
            ]


        # Patch "scan_nd" with dummy plan, don't need to test it again
        mocker.patch(
            "mx3_beamline_library.plans.basic_scans.scan_nd",
            side_effect=scan_nd_patched,
        )

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
                snake_axes=snake_axes,
                per_step=per_step,
                md=metadata,
            )
        )

        assert isinstance(res, tuple) and len(res) == 1
        assert isinstance(res[0], str) and UUID4(res[0])
