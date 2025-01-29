import pickle

import pytest
from bluesky.plan_stubs import null
from fakeredis import FakeStrictRedis
from pytest_mock.plugin import MockerFixture

from mx3_beamline_library.plans.xray_centering import XRayCentering
from mx3_beamline_library.schemas.xray_centering import (
    MD3ScanResponse,
    RasterGridCoordinates,
)


@pytest.fixture
def fake_redis() -> FakeStrictRedis:
    return FakeStrictRedis()


@pytest.fixture()
def x_ray_centering_instance(
    sample_id,
    mocker: MockerFixture,
    fake_redis: FakeStrictRedis,
    optical_centering_results,
):
    # Setup
    mocker.patch(
        "mx3_beamline_library.plans.xray_centering.redis_connection", new=fake_redis
    )
    fake_redis.set(
        f"optical_centering_results:{sample_id}",
        pickle.dumps(optical_centering_results),
    )

    return XRayCentering(
        sample_id=sample_id,
        grid_scan_id="edge",
        detector_distance=0.496,  # m
        photon_energy=13,  # keV
        omega_range=0,  # degrees
        md3_alignment_y_speed=10,  # mm/s
        count_time=None,
        hardware_trigger=True,
    )


def test_get_optical_centering_results(x_ray_centering_instance: XRayCentering):
    # Exercise
    result = x_ray_centering_instance.get_optical_centering_results()

    # Verify
    assert result is None


def test_get_optical_centering_results_failure(mocker, sample_id, fake_redis):
    mocker.patch(
        "mx3_beamline_library.plans.xray_centering.redis_connection", new=fake_redis
    )

    with pytest.raises(ValueError):
        result = XRayCentering(
            sample_id=sample_id,
            grid_scan_id="edge",
            detector_distance=0.496,  # m
            photon_energy=13,  # keV
            omega_range=0,  # degrees
            md3_alignment_y_speed=10,  # mm/s
            count_time=None,
            hardware_trigger=True,
        )
        result.get_optical_centering_results()


def test_calculate_md3_exposure_time(
    x_ray_centering_instance: XRayCentering, optical_centering_results
):
    # Setup
    grid = RasterGridCoordinates.model_validate(
        optical_centering_results["edge_grid_motor_coordinates"]
    )

    # exposure_time = grid.height_mm / md3_alignment_y_speed
    exposure_time = x_ray_centering_instance._calculate_md3_exposure_time(grid)
    assert exposure_time == pytest.approx(0.027, 0.2)


def test_calculate_md3_exposure_time_failure(
    optical_centering_results,
    sample_id,
    mocker,
    fake_redis,
):
    grid = RasterGridCoordinates.model_validate(
        optical_centering_results["edge_grid_motor_coordinates"]
    )
    mocker.patch(
        "mx3_beamline_library.plans.xray_centering.redis_connection", new=fake_redis
    )
    fake_redis.set(
        f"optical_centering_results:{sample_id}",
        pickle.dumps(optical_centering_results),
    )
    md3_alignment_y_seed = 20  # Set high speed

    with pytest.raises(ValueError):
        result = XRayCentering(
            sample_id=sample_id,
            grid_scan_id="edge",
            detector_distance=0.496,  # m
            photon_energy=13,  # keV
            omega_range=0,  # degrees
            md3_alignment_y_speed=md3_alignment_y_seed,  # mm/s
            count_time=None,
            hardware_trigger=True,
        )

        result._calculate_md3_exposure_time(grid)


def test_start_grid_scan(
    x_ray_centering_instance: XRayCentering,
    mocker: MockerFixture,
    optical_centering_results,
    run_engine,
):
    # Setup
    grid = RasterGridCoordinates.model_validate(
        optical_centering_results["edge_grid_motor_coordinates"]
    )
    grid_scan = mocker.patch(
        "mx3_beamline_library.plans.xray_centering.XRayCentering._grid_scan"
    )

    # Execute
    run_engine(x_ray_centering_instance.start_grid_scan())

    # Verify
    grid_scan.assert_called_once_with(grid)


def test_grid_scan(
    x_ray_centering_instance: XRayCentering,
    mocker: MockerFixture,
    optical_centering_results,
    run_engine,
):
    # Setup
    grid = RasterGridCoordinates.model_validate(
        optical_centering_results["edge_grid_motor_coordinates"]
    )

    scan_response = MD3ScanResponse(
        task_name="Raster Scan",
        task_flags=8,
        start_time="2023-02-21 12:40:47.502",
        end_time="2023-02-21 12:40:52.814",
        task_output="org.embl.dev.pmac.PmacDiagnosticInfo@64ba4055",
        task_exception="null",
        result_id=1,
    )

    def mock_slow_grid_scan(*args, **kwargs):
        yield from null()
        return scan_response

    grid_scan = mocker.patch(
        "mx3_beamline_library.plans.xray_centering.slow_grid_scan",
        side_effect=mock_slow_grid_scan,
    )

    # Execute
    run_engine(x_ray_centering_instance._grid_scan(grid))

    # Verify
    grid_scan.assert_called_once()
