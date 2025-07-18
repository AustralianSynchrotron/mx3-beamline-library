from os import path

import fakeredis
import numpy as np
import pytest
from bluesky import RunEngine
from pytest_mock.plugin import MockerFixture

from mx3_beamline_library.plans.image_analysis import (
    get_image_from_md3_camera,
    get_image_from_top_camera,
)
from mx3_beamline_library.plans.optical_centering import OpticalCentering, md3
from mx3_beamline_library.schemas.optical_centering import (
    OpticalCenteringExtraConfig,
    TopCamera,
)
from mx3_beamline_library.schemas.xray_centering import RasterGridCoordinates


def _set_redis_metadata(fake_redis):
    fake_redis.hset(
        "top_camera_target_coords",
        mapping={
            "x_pixel_target": 400,
            "y_pixel_target": 400,
        },
    )
    fake_redis.hset(
        "top_camera_pixels_per_mm",
        mapping={
            "pixels_per_mm_x": 40,
            "pixels_per_mm_y": 40,
        },
    )


@pytest.fixture
def fake_redis() -> fakeredis.FakeStrictRedis:
    return fakeredis.FakeStrictRedis()


@pytest.fixture
def optical_centering_instance(sample_id: str, fake_redis, mocker: MockerFixture):

    mocker.patch(
        "mx3_beamline_library.plans.optical_centering.redis_connection", fake_redis
    )
    _set_redis_metadata(fake_redis)
    optical_centering = OpticalCentering(
        sample_id=sample_id,
        beam_position=(612, 512),
        grid_step=(100, 100),
        plot=False,
        calibrated_alignment_z=0.47,
        manual_mode=False,
        use_top_camera_camera=True,
        extra_config=OpticalCenteringExtraConfig(
            top_camera=TopCamera(x_pixel_target=804, y_pixel_target=437)
        ),
    )

    return optical_centering


@pytest.mark.order("first")
def test_three_click_centering(
    optical_centering_instance: OpticalCentering, run_engine: RunEngine
):

    # Exercise
    run_engine(
        optical_centering_instance.three_click_centering(
            x_coords=[1, 1.4, 1.2],
            y_coords=[1, 1, 1],
            omega_positions=[0, np.pi / 2, np.pi],
        )
    )

    # Verify
    assert md3.sample_x.position == pytest.approx(0.3, 0.01)
    assert md3.sample_y.position == pytest.approx(-0.1, 0.01)
    assert md3.omega.position == pytest.approx(0, 0.01)
    assert md3.alignment_x.position == pytest.approx(0.43, 0.01)
    assert md3.alignment_y.position == pytest.approx(0.66, 0.01)
    assert md3.alignment_z.position == pytest.approx(-0.69, 0.01)


@pytest.mark.order(after="test_three_click_centering")
def test_two_click_centering(
    optical_centering_instance: OpticalCentering, run_engine: RunEngine
):
    # This test has to be executed after the three_click_centering test
    # because the md3 values change dynamically

    # Exercise
    run_engine(
        optical_centering_instance.two_click_centering(
            x_coords=[1, 1.4], y_coords=[1, 1], omega_positions=[0, np.pi / 2]
        )
    )

    # Verify
    assert md3.sample_x.position == pytest.approx(1.29, 0.01)
    assert md3.sample_y.position == pytest.approx(0.49, 0.5)
    assert md3.omega.position == pytest.approx(0, 0.01)
    assert md3.alignment_x.position == pytest.approx(0.43, 0.01)
    assert md3.alignment_y.position == pytest.approx(1.32, 0.01)
    assert md3.alignment_z.position == pytest.approx(0.47, 0.01)


def test_find_edge_and_flat_angles(
    optical_centering_instance: OpticalCentering,
    run_engine: RunEngine,
    mocker: MockerFixture,
):
    # Setup
    three_click_centering = mocker.patch(
        "mx3_beamline_library.plans.optical_centering.OpticalCentering.three_click_centering"
    )
    # Exercise
    run_engine(optical_centering_instance.find_edge_and_flat_angles())

    # Verify
    assert three_click_centering.call_count == 1


def test_center_loop(
    optical_centering_instance: OpticalCentering,
    mocker: MockerFixture,
    sample_id: str,
    fake_redis: fakeredis.FakeStrictRedis,
    run_engine: RunEngine,
):
    # Setup
    mocker.patch(
        "mx3_beamline_library.plans.optical_centering.redis_connection", new=fake_redis
    )
    # RE = RunEngine({})
    # bec = BestEffortCallback()
    # RE.subscribe(bec)

    # Exercise
    run_engine(optical_centering_instance.center_loop())

    # Verify
    assert fake_redis.get(f"optical_centering_results:{sample_id}") is not None


def test_center_loop_manual_mode(
    mocker: MockerFixture,
    sample_id: str,
    run_engine: RunEngine,
    fake_redis: fakeredis.FakeStrictRedis,
):
    # Setup
    mocker.patch(
        "mx3_beamline_library.plans.optical_centering.redis_connection", new=fake_redis
    )
    _set_redis_metadata(fake_redis)
    optical_centering = OpticalCentering(
        sample_id=sample_id,
        beam_position=(612, 512),
        grid_step=(100, 100),
        plot=False,
        calibrated_alignment_z=0.47,
        manual_mode=True,
        use_top_camera_camera=True,
        extra_config=OpticalCenteringExtraConfig(
            top_camera=TopCamera(x_pixel_target=804, y_pixel_target=437)
        ),
    )

    # Exercise
    run_engine(optical_centering.center_loop())

    # Verify
    assert fake_redis.get(f"optical_centering_results:{sample_id}") is None


def test_multipoint_centre_three_clicks(optical_centering_instance: OpticalCentering):
    # Exercise
    result = optical_centering_instance.multi_point_centre(
        x_coords=[1, 2, 3], omega_list=[0, 90, 180], two_clicks=False
    )

    # Verify
    assert round(result[0], 2) == -1.12
    assert round(result[1], 2) == 1.11


def test_multipoint_centre_two_clicks(optical_centering_instance: OpticalCentering):
    # Exercise
    result = optical_centering_instance.multi_point_centre(
        x_coords=[1, 2], omega_list=[0, 90], two_clicks=True
    )

    # Verify
    assert round(result[0], 2) == 2.16
    assert round(result[1], 2) == 0.28


def test_find_loop_edge_coordinates(optical_centering_instance: OpticalCentering):
    # Exercise
    result = optical_centering_instance.find_loop_edge_coordinates()

    # Verify
    assert result == (666, 168)


def test_prepare_raster_grid(optical_centering_instance: OpticalCentering):
    # Exercise
    result = optical_centering_instance.prepare_raster_grid(omega=0)

    # Verify
    assert isinstance(result, RasterGridCoordinates)


def test_get_md3_camera_jpeg_image(optical_centering_instance: OpticalCentering):
    # Exercise
    result = optical_centering_instance._get_md3_camera_jpeg_image()

    # Verify
    assert result is not None
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_sine_function(optical_centering_instance: OpticalCentering):
    # Setup and Exercise
    result = optical_centering_instance._sine_function(np.pi / 2, 1, -np.pi / 2, 0)

    # Verify
    assert result == 1.0


def test_three_click_centering_function(optical_centering_instance: OpticalCentering):
    # Setup and Exercise
    result = optical_centering_instance.three_click_centering_function(
        np.pi / 2, 1, 0, 0
    )

    # Verify
    assert result == 1.0


def test_two_click_centering_function(optical_centering_instance: OpticalCentering):
    # Setup and Exercise
    result = optical_centering_instance.two_click_centering_function(np.pi / 2, 1, 0)

    # Verify
    assert optical_centering_instance.beam_position == (612, 512)
    assert round(result, 2) == 1.41


def test_init_udc_error(fake_redis, mocker):
    # Exercise and verify
    _set_redis_metadata(fake_redis)
    mocker.patch(
        "mx3_beamline_library.plans.optical_centering.redis_connection", new=fake_redis
    )
    with pytest.raises(
        ValueError, match="grid_step can only be None if manual_mode=True"
    ):
        OpticalCentering(
            sample_id="sample_tmp",
            beam_position=(612, 512),
            grid_step=None,
            plot=False,
            calibrated_alignment_z=0.47,
            manual_mode=False,
            use_top_camera_camera=True,
            extra_config=OpticalCenteringExtraConfig(
                top_camera=TopCamera(x_pixel_target=804, y_pixel_target=437)
            ),
        )


@pytest.mark.order(after="test_two_click_centering")
def test_init_create_folder(sample_id: str, session_tmpdir, mocker, fake_redis):
    # Setup
    _set_redis_metadata(fake_redis)
    mocker.patch(
        "mx3_beamline_library.plans.optical_centering.redis_connection", new=fake_redis
    )
    OpticalCentering(
        sample_id=sample_id,
        beam_position=(612, 512),
        grid_step=(100, 100),
        plot=True,
        calibrated_alignment_z=0.47,
        manual_mode=False,
        use_top_camera_camera=True,
        output_directory=session_tmpdir,
    )

    # Verify
    output = path.join(session_tmpdir, sample_id)
    assert path.exists(output)


@pytest.mark.order(after="test_init_create_folder")
def test_find_loop_edge_coordinates_with_plot(
    sample_id: str, session_tmpdir, mocker, fake_redis
):
    # Setup
    _set_redis_metadata(fake_redis)
    mocker.patch(
        "mx3_beamline_library.plans.optical_centering.redis_connection", new=fake_redis
    )
    optical_centering = OpticalCentering(
        sample_id=sample_id,
        beam_position=(612, 512),
        grid_step=(100, 100),
        plot=True,
        calibrated_alignment_z=0.47,
        manual_mode=False,
        use_top_camera_camera=True,
        output_directory=session_tmpdir,
    )

    # Exercise
    result = optical_centering.find_loop_edge_coordinates()

    # Verify
    assert result == (666, 168)
    assert path.exists(
        path.join(
            session_tmpdir, sample_id, "test_sample_loop_centering_180_zoom_1.png"
        )
    )


@pytest.mark.order(after="test_find_loop_edge_coordinates_with_plot")
def test_find_zoom_0_maximum_area_with_plot(
    sample_id: str, session_tmpdir, run_engine: RunEngine, mocker, fake_redis
):
    # Setup
    _set_redis_metadata(fake_redis)
    mocker.patch(
        "mx3_beamline_library.plans.optical_centering.redis_connection", new=fake_redis
    )
    optical_centering = OpticalCentering(
        sample_id=sample_id,
        beam_position=(612, 512),
        grid_step=(100, 100),
        plot=True,
        calibrated_alignment_z=0.47,
        manual_mode=False,
        use_top_camera_camera=True,
        output_directory=session_tmpdir,
    )

    # Exercise
    run_engine(optical_centering._find_zoom_0_maximum_area())

    # Verify
    assert path.exists(
        path.join(session_tmpdir, sample_id, "test_sample_270_top_camera.png")
    )
    assert path.exists(
        path.join(session_tmpdir, sample_id, "test_sample_180_top_camera.png")
    )


@pytest.mark.order(after="test_find_zoom_0_maximum_area_with_plot")
def test_find_edge_and_flat_angles_with_plot(
    sample_id: str, session_tmpdir, run_engine: RunEngine, mocker, fake_redis
):
    # Setup
    _set_redis_metadata(fake_redis)
    mocker.patch(
        "mx3_beamline_library.plans.optical_centering.redis_connection", new=fake_redis
    )
    optical_centering = OpticalCentering(
        sample_id=sample_id,
        beam_position=(612, 512),
        grid_step=(100, 100),
        plot=True,
        calibrated_alignment_z=0.47,
        manual_mode=False,
        use_top_camera_camera=True,
        output_directory=session_tmpdir,
    )

    # Exercise
    run_engine(optical_centering.find_edge_and_flat_angles())

    # Verify
    assert path.exists(
        path.join(session_tmpdir, sample_id, f"{sample_id}_area_estimation_0.png")
    )
    assert path.exists(
        path.join(session_tmpdir, sample_id, f"{sample_id}_area_estimation_180.png")
    )
    assert path.exists(
        path.join(session_tmpdir, sample_id, f"{sample_id}_area_estimation_270.png")
    )


def test_prepare_raster_grid_with_plot(
    sample_id: str, session_tmpdir, mocker, fake_redis
):
    # Setup
    _set_redis_metadata(fake_redis)
    mocker.patch(
        "mx3_beamline_library.plans.optical_centering.redis_connection", new=fake_redis
    )
    optical_centering = OpticalCentering(
        sample_id=sample_id,
        beam_position=(612, 512),
        grid_step=(100, 100),
        plot=True,
        calibrated_alignment_z=0.47,
        manual_mode=False,
        use_top_camera_camera=True,
        output_directory=session_tmpdir,
    )
    filename = path.join(optical_centering.sample_path, f"{sample_id}_raster_grid_flat")

    # Exercise
    result = optical_centering.prepare_raster_grid(omega=0, filename=filename)

    # Verify
    assert isinstance(result, RasterGridCoordinates)
    assert path.exists(f"{filename}.png")


@pytest.mark.order(after="test_prepare_raster_grid_with_plot")
def test_save_gray_scale_image(sample_id: str, session_tmpdir, mocker, fake_redis):
    # Setup
    _set_redis_metadata(fake_redis)
    mocker.patch(
        "mx3_beamline_library.plans.optical_centering.redis_connection", new=fake_redis
    )
    optical_centering = OpticalCentering(
        sample_id=sample_id,
        beam_position=(612, 512),
        grid_step=(100, 100),
        plot=True,
        calibrated_alignment_z=0.47,
        manual_mode=False,
        use_top_camera_camera=True,
        output_directory=session_tmpdir,
    )
    filename = path.join(optical_centering.sample_path, "top_camera_plot")
    data, x_size, y_size = get_image_from_top_camera()
    new_data = data.reshape((y_size, x_size))
    # Exercise
    optical_centering.save_image(
        data=new_data, x_coord=1, y_coord=1, filename=filename, grayscale_img=True
    )

    # Verify
    assert path.exists(f"{filename}.png")


@pytest.mark.order(after="test_save_gray_scale_image")
def test_save_rgb_image(sample_id: str, session_tmpdir, mocker, fake_redis):
    # Setup
    _set_redis_metadata(fake_redis)
    mocker.patch(
        "mx3_beamline_library.plans.optical_centering.redis_connection", new=fake_redis
    )
    optical_centering = OpticalCentering(
        sample_id=sample_id,
        beam_position=(612, 512),
        grid_step=(100, 100),
        plot=True,
        calibrated_alignment_z=0.47,
        manual_mode=False,
        use_top_camera_camera=True,
        output_directory=session_tmpdir,
    )
    filename = path.join(optical_centering.sample_path, "md3_plot")
    data = get_image_from_md3_camera()
    # Exercise
    optical_centering.save_image(
        data=data, x_coord=1, y_coord=1, filename=filename, grayscale_img=False
    )

    # Verify
    assert path.exists(f"{filename}.png")
