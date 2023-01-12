import typing
import pytest
import imageio
from typing import Union
from numpy.typing import NDArray
from ophyd.signal import ConnectionTimeoutError
from mx3_beamline_library.plans.optical_centering import OpticalCentering
from mx3_beamline_library.science.optical_and_loop_centering.psi_optical_centering import (
    loopImageProcessing,
)

if typing.TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture
    from bluesky.utils import Msg
    from mx3_beamline_library.devices.sim import motors
    from mx3_beamline_library.devices.sim.classes.detectors import SimBlackFlyCam
    Motors = motors


@pytest.fixture(scope="function")
# @pytest.mark.parametrize("camera", ("sim_blackfly_camera",), indirect=True)
def optical_centering(motors: "Motors", camera: "SimBlackFlyCam") -> OpticalCentering:
    """Pytest fixture to initialise "OpticalCentering" for testing.

    Parameters
    ----------
    motors : Motors
        Loaded motors module, either simulated or real.
    camera : SimBlackFlyCam
        _description_

    Returns
    -------
    OpticalCentering
        OpticalCentering class instance.
    """

    optical_centering = OpticalCentering(
        camera,
        motors.testrig.x,
        motors.testrig.y,
        motors.testrig.z,
        motors.testrig.phi,
        beam_position=(640, 512),
        pixels_per_mm_x=292.87,
        pixels_per_mm_z=292.87,
        auto_focus=True,
        min_focus=-1,
        max_focus=0.5,
        tol=0.1,
        method="psi",
        plot=False,
    )

    yield optical_centering


@pytest.mark.parametrize("camera", ("sim_blackfly_camera",), indirect=True)
class TestOpticalCentering:
    """Run tests for Bluesky optical centering plans"""

    @pytest.mark.parametrize("image_dir", ("./tests/test_images/snapshot_6.jpg",))
    @pytest.mark.parametrize("as_gray", (False, True))
    def test_psi_optical_centering(self,
        optical_centering: OpticalCentering,
        image_dir: str,
        as_gray: bool,
    ):
        """ """

        width: int
        height: int
        snapshot = imageio.v2.imread(image_dir, as_gray=as_gray)
        width, height = snapshot.shape[:2]

        procImg = loopImageProcessing(image_dir)
        procImg.findContour(
            zoom=optical_centering.loop_img_processing_zoom,
            beamline=optical_centering.loop_img_processing_beamline,
        )

        # Check if "extremes" are within image width and height
        extremes = procImg.findExtremes()
        assert extremes["top"][0] >= 0 and extremes["top"][0] < height
        assert extremes["top"][1] >= 0 and extremes["top"][1] < width
        assert extremes["bottom"][0] >= 0 and extremes["bottom"][0] < height
        assert extremes["bottom"][1] >= 0 and extremes["bottom"][1] < width
        assert extremes["right"][0] >= 0 and extremes["right"][0] < height
        assert extremes["right"][1] >= 0 and extremes["right"][1] < width
        assert extremes["left"][0] >= 0 and extremes["left"][0] < height
        assert extremes["left"][1] >= 0 and extremes["left"][1] < width

        # Check if "rectangle_coordinates" is within image width and height
        rectangle_coordinates = procImg.fitRectangle()
        assert rectangle_coordinates[
            "top_left"
        ][0] >= 0 and rectangle_coordinates["top_left"][0] <= height
        assert rectangle_coordinates[
            "top_left"
        ][1] >= 0 and rectangle_coordinates["top_left"][1] <= width
        assert rectangle_coordinates[
            "bottom_right"
        ][0] >= 0 and rectangle_coordinates["bottom_right"][0] <= height
        assert rectangle_coordinates[
            "bottom_right"
        ][1] >= 0 and rectangle_coordinates["bottom_right"][1] <= width

    @pytest.mark.parametrize("auto_focus", (False, True))
    def test_center_loop(
        self,
        mocker: "MockerFixture",
        optical_centering: OpticalCentering,
        auto_focus: bool,
    ):
        """ """

        # Patch "auto_focus" property to pass selector for testing
        mocker.patch.object(optical_centering, "auto_focus", new=auto_focus)

        # Patch class methods covered by other tests
        mocker.patch("bluesky.plan_stubs.mv")
        unblur_image_patch = mocker.patch.object(optical_centering, "unblur_image")
        mocker.patch.object(optical_centering, "drive_motors_to_loop_edge")
        mocker.patch.object(optical_centering, "drive_motors_to_center_of_loop")

        # Call method to be tested
        tuple(optical_centering.center_loop())

        # Check that if "auto_focus" option is set the associated method is called
        if auto_focus:
            unblur_image_patch.assert_called_once()
        else:
            unblur_image_patch.assert_not_called()

    @pytest.mark.parametrize(
        ("a", "b", "tol", "final_variance", "optimal_value", "variances"),
        (
            (
                -1, 0.5, 0.1, 0.25, -0.958, (
                    50,
                    40,
                    30,
                    20,
                    10,
                    5,
                    4,
                    3,
                    2,
                    1,
                    0.5,
                    0.25,
                    0.125,
                    0.0625,
                    0.03125,
                ),
            ),
            (
                -1, 0.5, 0.5, -0.7, -0.823, (
                    -0.0625,
                    -0.125,
                    -0.25,
                    -0.5,
                    -0.6,
                    -0.7,
                    -0.8,
                    -0.9,
                    -1,
                    -1.1,
                    -1.2,
                    -1.3,
                    -1.4,
                    -1.5,
                )
            ),
            (
                -0.7, 0.5, 0.3, -0.7, -0.558, (
                    -0.0625,
                    -0.125,
                    -0.25,
                    -0.5,
                    -0.6,
                    -0.7,
                    -0.8,
                    -0.9,
                    -1,
                    -1.1,
                    -1.2,
                    -1.3,
                    -1.4,
                    -1.5,
                )
            ),
        )
    )
    @pytest.mark.timeout(10)
    def test_unblur_image(
        self,
        mocker: "MockerFixture",
        optical_centering: OpticalCentering,
        a: Union[int, float],
        b: Union[int, float],
        tol: Union[int, float],
        final_variance: Union[int, float],
        optimal_value: Union[int, float],
        variances: tuple[Union[float, int]],
    ):
        """ """

        # Patch "calculate_variance" method with generated list to test in isolation
        calculate_variance_patch = mocker.patch.object(
            optical_centering,
            "calculate_variance",
            side_effect=variances,
        )

        # Call method to test and get list of motor setpoints
        res: tuple[Msg] = tuple(
            filter(
                lambda msg: msg.command == "set",
                optical_centering.unblur_image(a, b, tol),
            )
        )

        # Check that the final returned variance fits what is expected
        assert final_variance == variances[calculate_variance_patch.call_count -1]

        # Check that the optimal motor setpoint fits what is expected
        assert optimal_value == round(res[-1][2][0], 3)

    @pytest.mark.parametrize(
        ("method", "method_exists"),
        (
            ("lucid3", True),
            ("psi", True),
            ("wrong", False),
        ),
    )
    @pytest.mark.parametrize(("x_coord", "y_coord"), ((125, 300), (245, 110)))
    @pytest.mark.parametrize("plot", (False, True))
    def test_drive_motors_to_loop_edge(
        self,
        mocker: "MockerFixture",
        optical_centering: OpticalCentering,
        method: str,
        method_exists: bool,
        x_coord: int,
        y_coord: int,
        plot: bool,
    ):
        """ """

        # Patch "method" property to pass selector for testing
        mocker.patch.object(optical_centering, "method", new=method)

        # Patch class methods covered by other tests
        mocker.patch.object(optical_centering, "get_image_from_camera")
        save_image_patch = mocker.patch.object(optical_centering, "save_image")
        mocker.patch.object(optical_centering, "plot", new=plot)

        # Patch "lucid3" class methods to test in isolation
        lucid_find_loop_patch = mocker.patch("lucid3.find_loop", return_value=(None, x_coord, y_coord))

        # Patch "loopImageProcessing" class methods to test in isolation
        mocker.patch.object(loopImageProcessing, "__init__", return_value=None)
        mocker.patch.object(loopImageProcessing, "findContour")
        psi_find_extremes_patch = mocker.patch.object(
            loopImageProcessing,
            "findExtremes",
            return_value={
                "bottom": (x_coord, y_coord),
            },
        )

        # Check that if the method does not exist it raises an exception
        if not method_exists:
            with pytest.raises(NotImplementedError) as e:
                tuple(optical_centering.drive_motors_to_loop_edge())
        else:
            # Call method to test and get list of motor setpoints
            res: tuple[Msg] = tuple(
                filter(
                    lambda msg: msg.command == "set",
                    optical_centering.drive_motors_to_loop_edge(),
                )
            )

            # Check that if "plot" option is set the "save_image" method is called
            if plot:
                save_image_patch.assert_called_once()
            else:
                save_image_patch.assert_not_called()

            # Check that the chosen "method" called the associated method
            if method == "lucid3":
                lucid_find_loop_patch.assert_called_once()
            elif method == "psi":
                psi_find_extremes_patch.assert_called_once()

            # Check that we only get two "set" messages and they are for different devices
            assert len(res) == 2
            assert res[0].obj != res[1].obj

            # Check that both "motor_x" and "motor_z" were given new setpoints
            for device, value in [(msg.obj, msg.args[0]) for msg in res]:
                assert device in (optical_centering.motor_x, optical_centering.motor_z)
                assert isinstance(value, float) or isinstance(value, int)

    @pytest.mark.parametrize(
        "rectangle_coordinates",
        (
            {
                "top_left": (882, 204),
                "bottom_right": (2048, 1231),
            },
        )
    )
    @pytest.mark.parametrize("plot", (False, True))
    def test_drive_motors_to_center_of_loop(
        self,
        mocker: "MockerFixture",
        optical_centering: OpticalCentering,
        rectangle_coordinates: dict[str, tuple[int]],
        plot: bool,
    ):
        """ """

        # Patch class methods covered by other tests
        mocker.patch.object(optical_centering, "plot", new=plot)
        mocker.patch.object(optical_centering, "get_image_from_camera")
        plot_raster_grid_and_center_of_loop_patch = mocker.patch.object(
            optical_centering,
            "plot_raster_grid_and_center_of_loop",
        )

        # Patch "loopImageProcessing" class methods to test in isolation
        mocker.patch.object(loopImageProcessing, "__init__", return_value=None)
        mocker.patch.object(loopImageProcessing, "findContour")
        mocker.patch.object(loopImageProcessing, "findExtremes")
        fit_rectangle_patch = mocker.patch.object(
            loopImageProcessing,
            "fitRectangle",
            return_value=rectangle_coordinates,
        )

        # Call method to test and get list of motor setpoints
        res: tuple[Msg] = tuple(
            filter(
                lambda msg: msg.command == "set",
                optical_centering.drive_motors_to_center_of_loop(),
            )
        )

        # Check that if "plot" option is set the associated method is called
        if plot:
            plot_raster_grid_and_center_of_loop_patch.assert_called_once()
        else:
            plot_raster_grid_and_center_of_loop_patch.assert_not_called()

        # Check that "findExtremes" method was called as expected
        fit_rectangle_patch.assert_called_once()

        # Check that we only get two "set" messages and they are for different devices
        assert len(res) == 2
        assert res[0].obj != res[1].obj

        # Check that both "motor_x" and "motor_z" were given new setpoints
        for device, value in [(msg.obj, msg.args[0]) for msg in res]:
            assert device in (optical_centering.motor_x, optical_centering.motor_z)
            assert isinstance(value, float) or isinstance(value, int)

    @pytest.mark.parametrize("image_dir", ("./tests/test_images/snapshot_6.jpg",))
    def test_calculate_variance(
        self,
        mocker: "MockerFixture",
        optical_centering: OpticalCentering,
        image_dir: str,
    ):
        """ """

        # Patch "get_image_from_camera" method to test in isolation
        mocker.patch.object(
            optical_centering,
            "get_image_from_camera",
            return_value=imageio.v2.imread(image_dir),
        )

        res = optical_centering.calculate_variance()

        assert res is not None and isinstance(res, float)

    @pytest.mark.parametrize("image_dir", ("./tests/test_images/snapshot_6.jpg",))
    @pytest.mark.parametrize("as_gray", (False, True))
    @pytest.mark.parametrize("fail_read_signal", (False, True))
    def test_get_image_from_camera(
        self,
        mocker: "MockerFixture",
        optical_centering: OpticalCentering,
        image_dir: str,
        as_gray: bool,
        fail_read_signal: bool,
    ):
        """ """

        # Set initial sim camera signals
        camera: "SimBlackFlyCam" = optical_centering.camera
        camera.set_values(imageio.v2.imread(image_dir, as_gray=as_gray))

        # Patch "numpy.load" and "logger" to test in isolation
        numpy_load_patch = mocker.patch("numpy.load")
        logger_info_patch = mocker.patch(
            "mx3_beamline_library.plans.optical_centering.logger.info",
        )

        if fail_read_signal:
            # Patch "array_data" signal "get" attribute to raise exception for the test
            mocker.patch.object(
                camera.array_data,
                "get",
                side_effect=ConnectionTimeoutError(),
            )

        # Call method to test
        data = optical_centering.get_image_from_camera()

        if fail_read_signal:
            # Check that the exception handler tried to load the backup static image
            numpy_load_patch.assert_called_once()

            # Check that we also logged a warning
            logger_info_patch.assert_called_once()
        else:
            # Check that data was returned as a Numpy array
            assert data is not None and isinstance(data, NDArray)

            # Check that array was reshapped as expected
            width, height = data.shape[:2]
            assert height == camera.width.get()
            assert width == camera.height.get()

    @pytest.mark.parametrize("image_dir", ("./tests/test_images/snapshot_6.jpg",))
    @pytest.mark.parametrize(("x_coord", "y_coord"), ((125, 300), (245, 110)))
    def test_save_image(
        self,
        mocker: "MockerFixture",
        optical_centering: OpticalCentering,
        image_dir: str,
        x_coord: int,
        y_coord: int,
    ):
        """ """

        filename = f"step_2_loop_centering_fig_{x_coord}"

        # Load test image data
        snapshot = imageio.v2.imread(image_dir)

        # Patch "savefig" since we don't actually want to save to disk
        savefig_patch = mocker.patch("matplotlib.pyplot.savefig")

        optical_centering.save_image(snapshot, x_coord, y_coord, filename)

        # Check that an attempt was made to save the plot
        savefig_patch.assert_called_once_with(filename)

    @pytest.mark.parametrize(
        ("image_dir", "rectangle_coordinates", "loop_center_coordinates"),
        (
            (
                "./tests/test_images/snapshot_6.jpg",
                {
                    "top_left": (882, 204),
                    "bottom_right": (2048, 1231),
                },
                (1465.0, 717.5),
            ),
        )
    )
    @pytest.mark.parametrize("filename", ("step_2_centered_loop",))
    def test_plot_raster_grid_and_center_of_loop(
        self,
        mocker: "MockerFixture",
        optical_centering: OpticalCentering,
        image_dir: str,
        rectangle_coordinates: dict[str, tuple[int]],
        loop_center_coordinates: tuple[float],
        filename: str,
    ):
        """ """

        # Set initial sim camera signals
        camera: "SimBlackFlyCam" = optical_centering.camera
        camera.set_values(imageio.v2.imread(image_dir))

        # Patch "savefig" since we don't actually want to save to disk
        savefig_patch = mocker.patch("matplotlib.pyplot.savefig")

        optical_centering.plot_raster_grid_and_center_of_loop(
            rectangle_coordinates,
            loop_center_coordinates,
            filename,
        )

        # Check that an attempt was made to save the plot
        savefig_patch.assert_called_once_with(filename)
