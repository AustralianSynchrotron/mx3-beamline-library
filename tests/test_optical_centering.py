import fakeredis
import pytest
from bluesky import RunEngine
from pytest_mock.plugin import MockerFixture

from mx3_beamline_library.plans.optical_centering import OpticalCentering 
from mx3_beamline_library.schemas.optical_centering import (
    OpticalCenteringExtraConfig,
    TopCamera,
)
from mx3_beamline_library.schemas.xray_centering import RasterGridCoordinates
import numpy as np
from mx3_beamline_library.plans.optical_centering import md3


@pytest.fixture
def fake_redis() -> fakeredis.FakeStrictRedis:
    return fakeredis.FakeStrictRedis()


@pytest.fixture
def optical_centering_instance(sample_id: str):
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
        optical_centering_instance: OpticalCentering,
    run_engine: RunEngine, sample_id: str):




    
    # Exercise
    run_engine(optical_centering_instance.three_click_centering(
        x_coords=[1, 1.4, 1.2], y_coords=[1,1,1], omega_positions=[0, 90, 180]
    ))

    # Verify
    assert round(md3.sample_x.position,2) == 0.14
    assert round(md3.sample_y.position,2) == -0.19
    assert round(md3.omega.position,2) == 0
    assert round(md3.alignment_x.position,2) == 0.43
    assert round(md3.alignment_y.position,2) == 0.66
    assert round(md3.alignment_z.position,2) == -0.78


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


def test_set_optical_centering_params(optical_centering_instance: OpticalCentering):
    # Setup
    config = OpticalCenteringExtraConfig()

    # Exercise
    optical_centering_instance._set_optical_centering_config_parameters(config)

    # Verify
    assert (
        optical_centering_instance.top_cam_pixels_per_mm_x
        == config.top_camera.pixels_per_mm_x
    )
    assert (
        optical_centering_instance.top_cam_pixels_per_mm_y
        == config.top_camera.pixels_per_mm_y
    )
    assert optical_centering_instance.x_pixel_target == config.top_camera.x_pixel_target
    assert optical_centering_instance.y_pixel_target == config.top_camera.y_pixel_target


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
    result = optical_centering_instance.prepare_raster_grid(
        omega=0)

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
    result = optical_centering_instance._sine_function(np.pi/2, 1, -np.pi/2, 0)

    # Verify
    assert result == 1.0

def test_three_click_centering_function(optical_centering_instance: OpticalCentering):
    # Setup and Exercise
    result = optical_centering_instance.three_click_centering_function(
        np.pi/2, 1, 0, 0
    )

    # Verify
    assert result == 1.0

def test_two_click_centering_function(optical_centering_instance: OpticalCentering):
    # Setup and Exercise
    result = optical_centering_instance.two_click_centering_function(
        np.pi/2, 1, 0
    )

    # Verify
    assert optical_centering_instance.beam_position == (612, 512)
    assert round(result,2) == 1.41


