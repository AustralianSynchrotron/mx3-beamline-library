import typing
import os
import pytest
import imageio
import numpy as np
import httpx
import pickle
from typing import Union, Optional
from numpy.typing import NDArray
from ophyd.signal import ConnectionTimeoutError
from requests_mock import Mocker as RequestMocker
from mx3_beamline_library.science.optical_and_loop_centering.crystal_finder import (
    CrystalFinder, CrystalFinder3D
)
from mx3_beamline_library.plans.optical_and_xray_centering import (
    OpticalAndXRayCentering, optical_and_xray_centering,
)
from mx3_beamline_library.science.optical_and_loop_centering.psi_optical_centering import (
    loopImageProcessing,
)
from mx3_beamline_library.schemas.optical_and_xray_centering import (
    RasterGridMotorCoordinates, SpotfinderResults, RasterGridMotorCoordinates,
)

if typing.TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture
    from bluesky.utils import Msg
    from mx3_beamline_library.devices.sim import motors, detectors
    from mx3_beamline_library.devices.sim.classes.detectors import (
        SimBlackFlyCam, SimDectrisDetector,
    )
    Motors = motors
    Detectors = detectors


@pytest.fixture(scope="function")
def xray_centering(
    motors: "Motors",
    detectors: "Detectors",
) -> OpticalAndXRayCentering:
    """Pytest fixture to initialise "OpticalAndXRayCentering" for testing.

    Parameters
    ----------
    motors : Motors
        Loaded motors module, either simulated or real.
    detectors : Detectors
        _description_

    Returns
    -------
    OpticalAndXRayCentering
        OpticalAndXRayCentering class instance.
    """

    xray_centering = OpticalAndXRayCentering(
        detectors.dectris_detector,
        detectors.sim_blackfly_camera,
        motors.testrig.x,
        motors.testrig.y,
        motors.testrig.z,
        motors.testrig.phi,
        md={"sample_id": "my_test_sample"},
        beam_position=[640, 512],
        pixels_per_mm_x=292.87,
        pixels_per_mm_z=292.87,
        threshold=20,
        auto_focus=False,
        min_focus=-1,
        max_focus=0.0,
        tol=0.1,
        method="psi",
        plot=True,
    )

    yield xray_centering


class TestXrayCentering:
    """Run tests for Bluesky xray centering plans"""

    @pytest.mark.parametrize(
        ("flat_path", "edge_path"),
        (
            (
                "./tests/crystal_finder/flat.npy",
                "./tests/crystal_finder/edge.npy",
            ),
        ),
    )
    @pytest.mark.parametrize("save", (False, True))
    @pytest.mark.parametrize("plot_cm", (False,))
    def test_crystal_finder(
        self,
        mocker: "MockerFixture",
        flat_path: str,
        edge_path: str,
        save: bool,
        plot_cm: bool,
    ):
        """ """

        # Patch "scatter3D" since
        scatter3D_patch = mocker.patch("mpl_toolkits.mplot3d.axes3d.Axes3D.scatter3D")

        # Patch "savefig" since we don't actually want to save to disk
        savefig_patch = mocker.patch("matplotlib.pyplot.savefig")

        # Load and prepare "flat" data into a Numpy array
        flat = np.load(flat_path)
        flat = np.rot90(np.append(flat, flat, axis=0), k=1)
        flat = np.append(flat, flat, axis=0)

        # Call the 2D crystal finder on "flat" data
        crystal_finder = CrystalFinder(flat, threshold=5)

        (
            cm_flat,
            coords_flat,
            distance_flat,
        ) = crystal_finder.plot_crystal_finder_results(save=save, filename="flat")

        # Check that if "save" option is set the associated method is called
        if save:
            savefig_patch.assert_called_once()

            # Reset mock call history
            savefig_patch.reset_mock()
        else:
            savefig_patch.assert_not_called()

        # Load and prepare "edge" data into a Numpy array
        edge = np.load(edge_path)
        edge = np.append(edge, edge, axis=1)

        # Call the 2D crystal finder on "edge" data
        crystal_finder = CrystalFinder(edge, threshold=5)

        (
            cm_edge,
            coords_edge,
            distance_edge,
        ) = crystal_finder.plot_crystal_finder_results(save=save, filename="edge")

        # Check that if "save" option is set the associated method is called
        if save:
            savefig_patch.assert_called_once()

            # Reset mock call history
            savefig_patch.reset_mock()
        else:
            savefig_patch.assert_not_called()

        # Call the 3D crystal finder with flat and edge results
        crystal_finder_3d = CrystalFinder3D(
            coords_flat, coords_edge, cm_flat, cm_edge, distance_flat, distance_edge
        )
        crystal_finder_3d.plot_crystals(plot_centers_of_mass=plot_cm, save=save)

        # Check that if "plot_cm" option is set the associated method is called
        if plot_cm:
            scatter3D_patch.assert_called_once()
        else:
            scatter3D_patch.assert_not_called()

        # Check that if "save" option is set the associated method is called
        if save:
            savefig_patch.assert_called_once()
        else:
            savefig_patch.assert_not_called()

    def test_start(
        self,
        mocker: "MockerFixture",
        xray_centering: OpticalAndXRayCentering,
    ):
        """ """

        def start_raster_scan_and_find_crystals_patched(*args, **kwargs):
            """ """

            yield None
            return (None, None, None, None)

        # Patch class methods covered by other tests
        mocker.patch.object(xray_centering, "center_loop")
        raster_scan_patch = mocker.patch.object(
            xray_centering,
            "start_raster_scan_and_find_crystals",
            side_effect=start_raster_scan_and_find_crystals_patched,
        )
        mocker.patch("bluesky.plan_stubs.mvr")
        mocker.patch.object(CrystalFinder3D, "__init__", return_value=None)
        plot_crystals_patch = mocker.patch.object(CrystalFinder3D, "plot_crystals")

        # Call method to test
        tuple(xray_centering.start())

        # Check that the raster scan start method was called twice
        raster_scan_patch.assert_called()
        assert raster_scan_patch.call_count == 2

        # Check that the 3D crystal finder was called
        plot_crystals_patch.assert_called_once()

    @pytest.mark.parametrize(
        "grid",
        (
            {
                "initial_pos_x": -1.290675043534674,
                "final_pos_x": 2.7111004882712466,
                "initial_pos_z": -0.2902311605831939,
                "final_pos_z": 1.2838460750503635,
            },
            {
                "initial_pos_x": -0.30730358179397,
                "final_pos_x": 2.7111004882712466,
                "initial_pos_z": -0.26291528664595215,
                "final_pos_z": 1.0482466623416533,
            },
        ),
    )
    @pytest.mark.parametrize("draw_grid_in_mxcube", (False, True))
    def test_start_raster_scan_and_find_crystals(
        self,
        mocker: "MockerFixture",
        xray_centering: OpticalAndXRayCentering,
        grid: dict,
        draw_grid_in_mxcube: bool,
    ):
        """ """

        # Patch "asyncio" module "get_event_loop" method to test in isolation
        mocker.patch("asyncio.get_event_loop")

        # Patch class methods covered by other tests
        mocker.patch.object(
            xray_centering,
            "prepare_raster_grid",
            return_value=(RasterGridMotorCoordinates.parse_obj(grid), {}),
        )
        draw_grid_in_mxcube_patch = mocker.patch.object(
            xray_centering,
            "draw_grid_in_mxcube",
        )
        mocker.patch.object(
            xray_centering,
            "find_crystal_positions",
            return_value=(None, None, None, None, None),
        )
        mocker.patch("mx3_beamline_library.plans.basic_scans.grid_scan")
        mocker.patch("bluesky.plan_stubs.mv")

        # Call method to test
        res = tuple(xray_centering.start_raster_scan_and_find_crystals(
            {
                "motor_x": xray_centering.motor_x.position,
                "motor_z": xray_centering.motor_z.position,
            },
            last_id=0,
            filename="flat",
            draw_grid_in_mxcube=draw_grid_in_mxcube,
        ))

        # Check that if draw grid option is set the associated method is called
        if draw_grid_in_mxcube:
            draw_grid_in_mxcube_patch.assert_called_once()
        else:
            draw_grid_in_mxcube_patch.assert_not_called()

    @pytest.mark.parametrize(
        "rectangle_coordinates",
        (
            {"top_left": (1018, 597), "bottom_right": (1434, 888)},
            {"top_left": (550, 589), "bottom_right": (1434, 819)},
        ),
    )
    @pytest.mark.parametrize("plot", (False, True))
    def test_prepare_raster_grid(
        self,
        mocker: "MockerFixture",
        xray_centering: OpticalAndXRayCentering,
        rectangle_coordinates: dict[str, tuple[int]],
        plot: bool,
    ):
        """ """

        # Patch class methods covered by other tests
        mocker.patch.object(xray_centering, "plot", new=plot)
        mocker.patch.object(xray_centering, "get_image_from_camera")
        plot_raster_grid_patch = mocker.patch.object(xray_centering, "plot_raster_grid")

        # Patch "loopImageProcessing" class methods to test in isolation
        mocker.patch.object(loopImageProcessing, "__init__", return_value=None)
        mocker.patch.object(loopImageProcessing, "findContour")
        mocker.patch.object(loopImageProcessing, "findExtremes")
        fit_rectangle_patch = mocker.patch.object(
            loopImageProcessing,
            "fitRectangle",
            return_value=rectangle_coordinates,
        )

        # Call method to test
        motor_coordinates, new_rect_coords = xray_centering.prepare_raster_grid()

        # Check that if "plot" option is set the associated method is called
        if plot:
            plot_raster_grid_patch.assert_called_once()
        else:
            plot_raster_grid_patch.assert_not_called()

        # Check that "findExtremes" method was called as expected
        fit_rectangle_patch.assert_called_once()

        # Check that returned "rectangle_coordinates" are as expected
        assert new_rect_coords == rectangle_coordinates

        # Check that returned "motor_coordinates" is a new model instance
        assert isinstance(motor_coordinates, RasterGridMotorCoordinates)

    @pytest.mark.parametrize(
        ("topic", "last_id", "npy_path", "n_rows", "n_cols", "centers_of_mass"),
        (
            (
                "flat",
                1,
                "./tests/crystal_finder/flat.npy",
                12,
                12,
                [(7, 7)],
            ),
            (
                "edge",
                2,
                "./tests/crystal_finder/edge.npy",
                12,
                4,
                [(3, 4), (0, 5), (3, 7), (0, 8), (3, 10), (0, 11)],
            ),
        ),
    )
    def test_find_crystal_positions(
        self,
        mocker: "MockerFixture",
        xray_centering: OpticalAndXRayCentering,
        topic: str,
        last_id: int,
        npy_path: str,
        n_rows: int,
        n_cols: int,
        centers_of_mass: list[tuple, int],
    ):
        """ """

        # Patch "CrystalFinder" class methds to test in isolation
        mocker.patch.object(CrystalFinder, "__init__", return_value=None)
        mocker.patch.object(
            CrystalFinder,
            "plot_crystal_finder_results",
            return_value=(centers_of_mass, [], []),
        )

        # Load spot finder results from file
        npy_array: NDArray = np.load(npy_path)
        number_of_spots_list = npy_array.tolist()

        # Build "SpotfinderResults" list from array
        spotfinder_results_list = []
        index = 1
        for row in number_of_spots_list:
            for n_spots in row:
                spotfinder_results_list.append(
                    (
                        SpotfinderResults(
                            type="start",
                            number_of_spots=n_spots,
                            image_id=index,
                            sequence_id=index,
                            bluesky_event_doc={},
                        ),
                        index,
                    )
                )
                index += 1

        # Patch Redis "xlen" method to lest in isolation
        mocker.patch.object(
            xray_centering.redis_connection,
            "xlen",
            return_value=len(spotfinder_results_list),
        )

        # Patch "read_message_from_redis_streams" method to return our test results
        mocker.patch.object(
            xray_centering,
            "read_message_from_redis_streams",
            side_effect=spotfinder_results_list,
        )

        # Call the method to test
        res = xray_centering.find_crystal_positions(topic, last_id, n_rows, n_cols)

        # Check that the data returned is in the expected format
        assert len(res) == 5
        spots_array, last_id = res[3:5]
        assert isinstance(spots_array, np.ndarray)
        assert last_id == len(spotfinder_results_list)

    @pytest.mark.parametrize(
        ("topic", "index", "data"),
        (
            (
                "edge",
                1,
                {
                    b"type": "start",
                    b"number_of_spots": 1125,
                    b"image_id": 1,
                    b"sequence_id": 1,
                    b"bluesky_event_doc": pickle.dumps(
                        {
                            "descriptor": "",
                            "time": 1673508496.587367,
                            "data": {
                                "dectris_detector_sequence_id": 1,
                                "testrig_x_user_setpoint": 0,
                                "testrig_x": 0,
                                "testrig_z_user_setpoint": 0,
                                "testrig_z": 0,
                            },
                            "timestamps": {},
                            "seq_num": 1,
                            "uid": "fe13ba1b-5703-4e49-8159-9bc1972fe431",
                            "filled": {},
                        }
                    ),
                },
            ),
            (
                "flat",
                2,
                {
                    b"type": "start",
                    b"number_of_spots": 1382,
                    b"image_id": 2,
                    b"sequence_id": 2,
                    b"bluesky_event_doc": pickle.dumps(
                        {
                            "descriptor": "",
                            "time": 1673508497.587367,
                            "data": {
                                "dectris_detector_sequence_id": 2,
                                "testrig_x_user_setpoint": 0,
                                "testrig_x": 0,
                                "testrig_z_user_setpoint": 0,
                                "testrig_z": 0,
                            },
                            "timestamps": {},
                            "seq_num": 1,
                            "uid": "4e44e602-5edc-4186-af29-b2e4421ad962",
                            "filled": {},
                        }
                    ),
                },
            ),
        ),
    )
    def test_read_message_from_redis_streams(
        self,
        mocker: "MockerFixture",
        xray_centering: OpticalAndXRayCentering,
        topic: str,
        data: dict,
        index: int,
    ):
        """ """

        # Patch Redis "xread" method to test in isolation
        mocker.patch.object(xray_centering.redis_connection, "xread", return_value=((None, ((index, data),)),))

        spotfinder_results, last_id = xray_centering.read_message_from_redis_streams(topic, index)

        # Check that "spotfinder_results" is new model instance of "SpotfinderResults"
        assert isinstance(spotfinder_results, SpotfinderResults)

        # Check that "last_id" is as expected
        assert last_id == index

    @pytest.mark.parametrize(
        "rectangle_coordinates",
        (
            {
                "top_left": np.asarray((882, 204)),
                "bottom_right": np.asarray((2048, 1231)),
            },
        ),
    )
    @pytest.mark.parametrize(
        "number_of_spots_array",
        (None, np.empty(3, dtype=np.int8)),
    )
    @pytest.mark.parametrize(
        ("num_cols", "num_rows"),
        (
            (6, 6),
        ),
    )
    @pytest.mark.parametrize("connection_error", (False, True))
    @pytest.mark.asyncio
    async def test_draw_grid_in_mxcube(
        self,
        mocker: "MockerFixture",
        xray_centering: OpticalAndXRayCentering,
        rectangle_coordinates: dict[str, tuple[int]],
        number_of_spots_array: Optional[NDArray],
        num_cols: int,
        num_rows: int,
        connection_error: bool,
    ):
        """ """

        # Patch class methods covered by other tests
        mocker.patch.object(
            xray_centering,
            "create_heatmap_and_crystal_map",
            return_value=dict(),
        )

        # Patch "AsyncClient.post" to test in isolation
        # We're not using the "requests_mock" fixture here as won't work with async
        client_post_patch = mocker.patch.object(
            httpx.AsyncClient,
            "post",
            return_value=httpx.Response(
                status_code=200,
                text="Successful test request.",
            ),
        )

        if connection_error:
            # Patch "AsyncClient.__init__" to raise an exception
            async_client_patch = mocker.patch.object(
                httpx.AsyncClient,
                "__init__",
                side_effect=httpx.ConnectError("Test Connection issue."),
            )

        # Await call method to test
        await xray_centering.draw_grid_in_mxcube(
            rectangle_coordinates,
            num_cols,
            num_rows,
            number_of_spots_array,
        )

        if not connection_error:
            # Check that an attempt was made to call the MXCuBE endpoint
            client_post_patch.assert_called_once()
        else:
            # Check that an attempt was made to call the async client
            async_client_patch.assert_called_once()

            # Check that no attempt was made to call the MXCuBE endpoint
            client_post_patch.assert_not_called()

    @pytest.mark.parametrize(
        ("num_cols", "num_rows", "number_of_spots_array"),
        (
            (
                4,
                12,
                np.asarray(
                    [
                        [0, 1, 2, 0],
                        [1, 0, 2, 14],
                        [11, 1, 0, 0],
                        [1, 4, 4, 4],
                        [3, 1, 16, 36],
                        [35, 0, 1, 1],
                        [1, 1, 0, 0],
                        [2, 3, 362, 548],
                        [345, 8, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 201, 355],
                        [504, 1, 0, 0],
                    ],
                    # dtype=np.int8,
                ),
            ),
            (
                12,
                12,
                np.asarray(
                    [
                        [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
                        [1, 0, 1, 0, 3, 1, 0, 0, 0, 1, 0, 0],
                        [0, 0, 2, 0, 1, 1, 1, 0, 0, 1, 1, 0],
                        [0, 0, 1, 1, 0, 0, 1, 3, 1, 1, 1, 0],
                        [0, 0, 1, 1, 0, 0, 6, 16, 13, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 18, 57, 52, 1, 0, 0],
                        [1, 3, 1, 1, 0, 9, 447, 496, 466, 7, 0, 1],
                        [0, 0, 0, 0, 0, 0, 336, 497, 477, 51, 0, 2],
                        [0, 0, 1, 0, 0, 0, 1, 191, 114, 1, 0, 1],
                        [1, 0, 0, 0, 1, 2, 0, 0, 1, 1, 1, 2],
                        [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    ],
                    # dtype=np.int8,
                ),
            ),
        ),
    )
    def test_create_heatmap_and_crystal_map(
        self,
        xray_centering: OpticalAndXRayCentering,
        num_cols: int,
        num_rows: int,
        number_of_spots_array: NDArray,
    ):
        """ """

        res = xray_centering.create_heatmap_and_crystal_map(
            num_cols,
            num_rows,
            number_of_spots_array,
        )

        # Check output is in the correct format
        assert res is not None and isinstance(res, dict)
        assert res.get("heatmap") is not None and isinstance(res["heatmap"], dict)
        assert res.get("crystalmap") is not None and isinstance(res["crystalmap"], dict)

    @pytest.mark.parametrize(
        "rectangle_coordinates",
        (
            {
                "top_left": (882, 204),
                "bottom_right": (2048, 1231),
            },
        ),
    )
    @pytest.mark.parametrize("image_dir", ("./tests/test_images/snapshot_6.jpg",))
    def test_plot_raster_grid(
        self,
        mocker: "MockerFixture",
        xray_centering: OpticalAndXRayCentering,
        rectangle_coordinates: dict[str, tuple[int]],
        image_dir: str,
    ):
        """ """

        # Patch class methods covered by other tests
        mocker.patch.object(
            xray_centering,
            "get_image_from_camera",
            return_value=imageio.v2.imread(image_dir),
        )

        # Patch "savefig" since we don't actually want to save to disk
        savefig_patch = mocker.patch("matplotlib.pyplot.savefig")

        xray_centering.plot_raster_grid(rectangle_coordinates, "test_plot")

        # Check that an attempt was made to save the plot
        savefig_patch.assert_called_once()

    def test_optical_and_xray_centering(
        self,
        mocker: "MockerFixture",
        xray_centering: OpticalAndXRayCentering,
    ):
        """ """

        # Patch class methods covered by other tests
        mocker.patch.object(OpticalAndXRayCentering, "__init__", return_value=None)
        start_patch = mocker.patch.object(OpticalAndXRayCentering, "start")

        tuple(optical_and_xray_centering(
            xray_centering.detector,
            xray_centering.camera,
            xray_centering.motor_x,
            xray_centering.motor_y,
            xray_centering.motor_z,
            xray_centering.motor_phi,
            xray_centering.md,
            xray_centering.beam_position,
            xray_centering.beam_size,
            xray_centering.pixels_per_mm_x,
            xray_centering.pixels_per_mm_z,
        ))

        # Check that "start" method was called
        start_patch.assert_called_once()
