import json

import httpx
import numpy as np
import respx
from pytest_mock.plugin import MockerFixture

from mx3_beamline_library.devices.detectors import dectris_detector
from mx3_beamline_library.devices.motors import md3
from mx3_beamline_library.plans.basic_scans import (
    _calculate_alignment_y_motor_coords,
    _calculate_alignment_z_motor_coords,
    _calculate_sample_x_coords,
    _calculate_sample_y_coords,
    md3_4d_scan,
    md3_grid_scan,
    md3_scan,
    slow_grid_scan,
)
from mx3_beamline_library.schemas.crystal_finder import MotorCoordinates
from mx3_beamline_library.schemas.detector import UserData
from mx3_beamline_library.schemas.xray_centering import RasterGridCoordinates


@respx.mock(assert_all_mocked=False)
def test_md3_scan(respx_mock, run_engine, sample_id, mocker: MockerFixture):
    # Setup
    arm = respx_mock.put("http://0.0.0.0:8000/detector/api/1.8.0/command/arm").mock(
        return_value=httpx.Response(200, content=json.dumps({"sequence id": 1}))
    )
    beam_center = mocker.patch("mx3_beamline_library.plans.basic_scans.set_beam_center")
    mocker.patch("mx3_beamline_library.plans.beam_utils.redis_connection")
    mocker.patch(
        "mx3_beamline_library.plans.basic_scans.save_screen_or_dataset_crystal_pic_to_redis"
    )

    screening = md3_scan(
        id=sample_id,
        crystal_id=1,
        number_of_frames=1,
        scan_range=1,
        exposure_time=2,
        data_collection_id=0,
        photon_energy=13.0,
        detector_distance=0.4,
        transmission=0.1,
        tray_scan=False,
    )

    # Exercise
    run_engine(screening)

    # Verify
    assert arm.call_count == 1
    beam_center.assert_called_once_with(0.4 * 1000)


@respx.mock(assert_all_mocked=False)
def test_md3_tray_scan(respx_mock, run_engine, sample_id, mocker: MockerFixture):
    # Setup
    arm = respx_mock.put("http://0.0.0.0:8000/detector/api/1.8.0/command/arm").mock(
        return_value=httpx.Response(200, content=json.dumps({"sequence id": 1}))
    )
    beam_center = mocker.patch("mx3_beamline_library.plans.basic_scans.set_beam_center")

    mocker.patch("mx3_beamline_library.plans.beam_utils.set_beam_center")
    mocker.patch("mx3_beamline_library.plans.beam_utils.redis_connection")
    mocker.patch(
        "mx3_beamline_library.plans.basic_scans.save_screen_or_dataset_crystal_pic_to_redis"
    )

    screening = md3_scan(
        id=sample_id,
        crystal_id=1,
        number_of_frames=1,
        scan_range=1,
        exposure_time=2,
        data_collection_id=0,
        photon_energy=13.0,
        transmission=0.1,
        detector_distance=0.4,
        tray_scan=True,
        motor_positions=MotorCoordinates(
            sample_x=0,
            sample_y=0,
            alignment_x=0,
            alignment_y=0,
            alignment_z=0,
            omega=90,
        ),
        drop_location="A1-1",
    )

    # Exercise
    run_engine(screening)

    # Verify
    assert arm.call_count == 1
    beam_center.assert_called_once_with(0.4 * 1000)


@respx.mock(assert_all_mocked=False)
def test_md3_grid_scan(respx_mock, run_engine, mocker: MockerFixture):
    # Setup
    arm = respx_mock.put("http://0.0.0.0:8000/detector/api/1.8.0/command/arm").mock(
        return_value=httpx.Response(200, content=json.dumps({"sequence id": 1}))
    )
    beam_center = mocker.patch("mx3_beamline_library.plans.basic_scans.set_beam_center")
    mocker.patch("mx3_beamline_library.plans.beam_utils.redis_connection")
    mocker.patch("mx3_beamline_library.plans.basic_scans.SERVER")
    task_info = mocker.patch(
        "mx3_beamline_library.plans.basic_scans.SERVER.retrieveTaskInfo",
        return_value=[
            "",
            1,
            "",
            "",
            "",
            "null",
            1,
        ],
    )

    user_data = UserData(
        id="my_sample",
        collection_type="grid_scan",
        grid_scan_id="flat",
    )

    # Exercise
    run_engine(
        md3_grid_scan(
            detector=dectris_detector,
            grid_width=0.7839332119645885,
            grid_height=0.49956528213429663,
            number_of_columns=4,
            number_of_rows=4,
            start_omega=176.7912087912088,
            omega_range=0,
            start_alignment_y=-0.010115684286529321,
            start_alignment_z=0.6867517681659011,
            start_sample_x=-0.10618655152995649,
            start_sample_y=-0.4368335669982139,
            md3_exposure_time=1,
            user_data=user_data,
            detector_distance=0.4,
            photon_energy=13,
            transmission=0.1,
        )
    )

    # Verify
    assert arm.call_count == 1
    assert task_info.call_count == 1
    beam_center.assert_called_once_with(0.4 * 1000)


@respx.mock(assert_all_mocked=False)
def test_md3_4d_scan(respx_mock, run_engine, mocker: MockerFixture):
    # Setup
    arm = respx_mock.put("http://0.0.0.0:8000/detector/api/1.8.0/command/arm").mock(
        return_value=httpx.Response(200, content=json.dumps({"sequence id": 1}))
    )
    mocker.patch("mx3_beamline_library.plans.basic_scans.SERVER")
    mocker.patch(
        "mx3_beamline_library.plans.basic_scans.set_distance_phase_and_transmission"
    )
    beam_center = mocker.patch("mx3_beamline_library.plans.basic_scans.set_beam_center")

    task_count = mocker.patch(
        "mx3_beamline_library.plans.basic_scans.SERVER.retrieveTaskInfo",
        return_value=[
            "",
            1,
            "",
            "",
            "",
            "null",
            1,
        ],
    )
    user_data = UserData(
        id="my_sample",
        collection_type="grid_scan",
        grid_scan_id="flat",
    )

    # Exercise
    run_engine(
        md3_4d_scan(
            detector=dectris_detector,
            start_angle=176.7912087912088,
            scan_range=0,
            md3_exposure_time=2,
            start_alignment_y=-0.010115684286529321,
            stop_alignment_y=0.57,
            start_sample_x=-0.10618655152995649,
            stop_sample_x=-0.10618655152995649,
            start_sample_y=-0.106,
            stop_sample_y=-0.106,
            start_alignment_z=1.1,
            stop_alignment_z=1.1,
            number_of_frames=8,
            user_data=user_data,
            detector_distance=0.4,
            photon_energy=13,
            transmission=0.1,
        )
    )

    # Verify
    assert arm.call_count == 1
    assert task_count.call_count == 1
    beam_center.assert_called_once_with(0.4 * 1000)


@respx.mock(assert_all_mocked=False)
def test_slow_grid_scan(respx_mock, run_engine, optical_centering_results):
    # Setup
    arm = respx_mock.put("http://0.0.0.0:8000/detector/api/1.8.0/command/arm").mock(
        return_value=httpx.Response(200, content=json.dumps({"sequence id": 1}))
    )
    grid = RasterGridCoordinates.model_validate(
        optical_centering_results["edge_grid_motor_coordinates"]
    )

    # Exercise
    run_engine(
        slow_grid_scan(
            raster_grid_coords=grid,
            detector=dectris_detector,
            detector_configuration={},
            alignment_y=md3.alignment_y,
            alignment_z=md3.alignment_z,
            sample_x=md3.sample_x,
            sample_y=md3.sample_y,
            omega=md3.omega,
            use_centring_table=True,
        )
    )

    # Verify
    assert arm.call_count == 1


@respx.mock(assert_all_mocked=False)
def test_slow_grid_scan_no_centring_table(
    respx_mock, run_engine, optical_centering_results
):
    # Setup
    arm = respx_mock.put("http://0.0.0.0:8000/detector/api/1.8.0/command/arm").mock(
        return_value=httpx.Response(200, content=json.dumps({"sequence id": 1}))
    )
    grid = RasterGridCoordinates.model_validate(
        optical_centering_results["edge_grid_motor_coordinates"]
    )

    # Exercise
    run_engine(
        slow_grid_scan(
            raster_grid_coords=grid,
            detector=dectris_detector,
            detector_configuration={},
            alignment_y=md3.alignment_y,
            alignment_z=md3.alignment_z,
            sample_x=md3.sample_x,
            sample_y=md3.sample_y,
            omega=md3.omega,
            use_centring_table=False,
        )
    )

    # Verify
    assert arm.call_count == 1


def test_calculate_alignment_y_motor_coords(optical_centering_results):
    # Setup
    grid = RasterGridCoordinates.model_validate(
        optical_centering_results["edge_grid_motor_coordinates"]
    )

    # Exercise
    result = _calculate_alignment_y_motor_coords(grid)

    # Verify
    assert np.array_equal(
        np.round(result, 2), np.array([[1.11, 1.38], [1.25, 1.25], [1.38, 1.11]])
    )


def test_calculate_alignment_z_motor_coords(optical_centering_results):
    # Setup
    grid = RasterGridCoordinates.model_validate(
        optical_centering_results["edge_grid_motor_coordinates"]
    )

    # Exercise
    result = _calculate_alignment_z_motor_coords(grid)

    # Verify
    assert np.array_equal(
        np.round(result, 2), np.array([[0.43, 0.43], [0.43, 0.43], [0.43, 0.43]])
    )


def test_calculate_sample_x_coords(optical_centering_results):
    # Setup
    grid = RasterGridCoordinates.model_validate(
        optical_centering_results["edge_grid_motor_coordinates"]
    )

    # Exercise
    result = _calculate_sample_x_coords(grid)

    # Verify
    assert np.array_equal(
        np.round(result, 3), np.array([[0.133, 0.007], [0.133, 0.007], [0.133, 0.007]])
    )


def test_calculate_sample_y_coords(optical_centering_results):
    # Setup
    grid = RasterGridCoordinates.model_validate(
        optical_centering_results["edge_grid_motor_coordinates"]
    )

    # Exercise
    result = _calculate_sample_y_coords(grid)

    # Verify
    assert np.array_equal(
        np.round(result, 3),
        np.array([[-7.158, -7.158], [-7.158, -7.158], [-7.158, -7.158]]),
    )
