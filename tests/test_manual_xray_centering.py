import pytest
from pytest_mock.plugin import MockerFixture

from mx3_beamline_library.plans.manual_xray_centering import ManualXRayCentering
from mx3_beamline_library.schemas.crystal_finder import MotorCoordinates


@pytest.fixture()
def xray_centering_instance(sample_id) -> ManualXRayCentering:
    return ManualXRayCentering(
        sample_id=sample_id,
        grid_scan_id="manual_collection",
        grid_top_left_coordinate=(388, 502),
        grid_height=260,
        grid_width=364,
        beam_position=(612, 512),
        number_of_columns=7,
        number_of_rows=5,
        detector_distance=0.496,  # m
        photon_energy=13,  # keV
        transmission=0.1,
        omega_range=0,  # degrees
        md3_alignment_y_speed=1,  # mm/s
        count_time=None,
        hardware_trigger=True,
    )


@pytest.mark.order("first")
def test_prepare_raster_grid_coordinates(xray_centering_instance: ManualXRayCentering):
    # Exercise
    result = xray_centering_instance.prepare_raster_grid(omega=0)

    # Verify
    assert result.initial_pos_sample_x == pytest.approx(0, 0.01)
    assert result.final_pos_sample_x == pytest.approx(0, 0.01)
    assert result.initial_pos_sample_y == pytest.approx(-0.15, 0.01)
    assert result.final_pos_sample_y == pytest.approx(0.093, 0.01)
    assert result.initial_pos_alignment_y == pytest.approx(-0.0066, 0.1)
    assert result.final_pos_alignment_y == pytest.approx(0.166, 0.01)
    assert result.initial_pos_alignment_z == pytest.approx(0.0, 0.01)
    assert result.final_pos_alignment_z == pytest.approx(0.0, 0.01)
    assert result.pixels_per_mm == 1500.0
    assert result.width_mm == pytest.approx(0.242, 0.01)
    assert result.height_mm == pytest.approx(0.173, 0.01)
    assert result.top_left_pixel_coordinates == (388, 502)
    assert result.bottom_right_pixel_coordinates == (752, 762)
    assert result.width_pixels == 364
    assert result.height_pixels == 260
    assert result.number_of_columns == 7
    assert result.number_of_rows == 5


def test_get_optical_centering_results(xray_centering_instance: ManualXRayCentering):
    # Exercise
    result = xray_centering_instance.get_optical_centering_results()

    # Verify
    assert result is None


def test_get_current_motor_positions(xray_centering_instance: ManualXRayCentering):
    # Exercise
    result = xray_centering_instance._get_current_motor_positions()

    # Verify
    assert isinstance(result, MotorCoordinates)


def test_start_grid_scan(
    xray_centering_instance: ManualXRayCentering, run_engine, mocker: MockerFixture
):
    # Setup
    grid_scan = mocker.patch(
        "mx3_beamline_library.plans.manual_xray_centering.ManualXRayCentering._grid_scan"
    )
    mocker.patch("mx3_beamline_library.plans.manual_xray_centering.redis_connection")

    # Exercise
    run_engine(xray_centering_instance.start_grid_scan())

    # Verify
    grid_scan.assert_called_once()
