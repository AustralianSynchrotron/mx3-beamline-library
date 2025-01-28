import pytest
from bluesky.plan_stubs import null
from fakeredis import FakeStrictRedis
from pytest_mock.plugin import MockerFixture

from mx3_beamline_library.plans.tray_scans import (
    multiple_drop_grid_scan,
    single_drop_grid_scan,
)
from mx3_beamline_library.schemas.xray_centering import MD3ScanResponse


@pytest.fixture
def fake_redis() -> FakeStrictRedis:
    return FakeStrictRedis()


def mock_slow_grid_scan(*args, **kwargs):
    yield from null()
    return MD3ScanResponse(
        task_name="Raster Scan",
        task_flags=8,
        start_time="2023-02-21 12:40:47.502",
        end_time="2023-02-21 12:40:52.814",
        task_output="org.embl.dev.pmac.PmacDiagnosticInfo@64ba4055",
        task_exception="null",
        result_id=1,
    )


def test_single_drop_grid_scan(
    run_engine, sample_id, mocker: MockerFixture, fake_redis
):
    # Setup
    grid_scan = mocker.patch(
        "mx3_beamline_library.plans.tray_scans.slow_grid_scan",
        side_effect=mock_slow_grid_scan,
    )
    mocker.patch(
        "mx3_beamline_library.plans.tray_scans.redis_connection", new=fake_redis
    )
    drop_location = "A2-1"

    # Exercise
    run_engine(
        single_drop_grid_scan(
            tray_id=sample_id,
            drop_location="A2-1",
            detector_distance=0.264,
            photon_energy=13,
            grid_number_of_columns=70,
            grid_number_of_rows=70,
            md3_alignment_y_speed=10,
            omega_range=0,
            alignment_y_offset=0,
            alignment_z_offset=0,
        )
    )

    # Verify
    grid_scan.assert_called_once()
    assert (
        fake_redis.get(f"tray_raster_grid_coordinates_{drop_location}:{sample_id}")
        is not None
    )


def test_multiple_drop_grid_scan(
    run_engine, sample_id, mocker: MockerFixture, fake_redis
):
    # Setup

    grid_scan = mocker.patch(
        "mx3_beamline_library.plans.tray_scans.slow_grid_scan",
        side_effect=mock_slow_grid_scan,
    )
    mocker.patch(
        "mx3_beamline_library.plans.tray_scans.redis_connection", new=fake_redis
    )
    drop_locations = ["A2-1", "A1-1"]

    # Exercise
    run_engine(
        multiple_drop_grid_scan(
            tray_id=sample_id,
            drop_locations=["A2-1", "A1-1"],
            detector_distance=0.264,
            photon_energy=13,
            grid_number_of_columns=70,
            grid_number_of_rows=70,
            md3_alignment_y_speed=10,
            omega_range=0,
            alignment_y_offset=0,
            alignment_z_offset=0,
        )
    )

    # Verify
    assert grid_scan.call_count == 2
    assert (
        fake_redis.get(f"tray_raster_grid_coordinates_{drop_locations[0]}:{sample_id}")
        is not None
    )
    assert (
        fake_redis.get(f"tray_raster_grid_coordinates_{drop_locations[1]}:{sample_id}")
        is not None
    )


def test_multiple_drop_grid_scan_frame_rate_error(
    run_engine, sample_id, mocker: MockerFixture, fake_redis
):
    # Setup

    grid_scan = mocker.patch(
        "mx3_beamline_library.plans.tray_scans.slow_grid_scan",
        side_effect=mock_slow_grid_scan,
    )
    mocker.patch(
        "mx3_beamline_library.plans.tray_scans.redis_connection", new=fake_redis
    )
    drop_locations = ["A2-1", "A1-1"]
    md3_alignment_y_speed = 18  # This triggers the frame rate error

    # Exercise and verify
    with pytest.raises(ValueError):
        run_engine(
            multiple_drop_grid_scan(
                tray_id=sample_id,
                drop_locations=drop_locations,
                detector_distance=0.264,
                photon_energy=13,
                grid_number_of_columns=70,
                grid_number_of_rows=70,
                md3_alignment_y_speed=md3_alignment_y_speed,
                omega_range=0,
                alignment_y_offset=0,
                alignment_z_offset=0,
            )
        )
    grid_scan.assert_not_called()


def test_multiple_drop_grid_scan_width_error(
    run_engine, sample_id, mocker: MockerFixture, fake_redis
):
    # Setup
    grid_scan = mocker.patch(
        "mx3_beamline_library.plans.tray_scans.slow_grid_scan",
        side_effect=mock_slow_grid_scan,
    )
    mocker.patch(
        "mx3_beamline_library.plans.tray_scans.redis_connection", new=fake_redis
    )
    drop_locations = ["A2-1", "A1-1"]
    grid_number_of_columns = 1  # This triggers the width error

    # Exercise and verify
    with pytest.raises(ValueError):
        run_engine(
            multiple_drop_grid_scan(
                tray_id=sample_id,
                drop_locations=drop_locations,
                detector_distance=0.264,
                photon_energy=13,
                grid_number_of_columns=grid_number_of_columns,
                grid_number_of_rows=70,
                md3_alignment_y_speed=13,
                omega_range=0,
                alignment_y_offset=0,
                alignment_z_offset=0,
            )
        )
    grid_scan.assert_not_called()
