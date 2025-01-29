from mx3_beamline_library.plans.basic_scans import md3_scan, md3_grid_scan, md3_4d_scan
from mx3_beamline_library.devices.motors import md3
from mx3_beamline_library.devices.detectors import dectris_detector
from mx3_beamline_library.schemas.detector import UserData  # noqa

import respx
import httpx
import json
from mx3_beamline_library.schemas.crystal_finder import MotorCoordinates
import pytest
from bluesky.plan_stubs import mv
from pytest_mock.plugin import MockerFixture
from mx3_beamline_library.schemas.xray_centering import MD3ScanResponse

@respx.mock(assert_all_mocked=False)
def test_md3_scan(respx_mock, run_engine, sample_id):
    arm = respx_mock.put("http://0.0.0.0:8000/detector/api/1.8.0/command/arm").mock(return_value=httpx.Response(200, content=json.dumps({"sequence id": 1})))
    screening = md3_scan(
        id=sample_id,
        crystal_id=1,
        number_of_frames=1,
        scan_range=1,
        exposure_time=2,
        data_collection_id=0,
        photon_energy=13.0,
        detector_distance=0.4,
        tray_scan=False,
    )
    run_engine(screening)

    assert arm.call_count == 1



@respx.mock(assert_all_mocked=False)
def test_md3_tray_scan(respx_mock, run_engine, sample_id):
    arm = respx_mock.put("http://0.0.0.0:8000/detector/api/1.8.0/command/arm").mock(return_value=httpx.Response(200, content=json.dumps({"sequence id": 1})))
    screening = md3_scan(
        id=sample_id,
        crystal_id=1,
        number_of_frames=1,
        scan_range=1,
        exposure_time=2,
        data_collection_id=0,
        photon_energy=13.0,
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
        drop_location="A1-1"
    )
    run_engine(screening)

    assert arm.call_count == 1

@respx.mock(assert_all_mocked=False)
def test_md3_grid_scan(respx_mock, run_engine, mocker: MockerFixture):
    arm = respx_mock.put("http://0.0.0.0:8000/detector/api/1.8.0/command/arm").mock(return_value=httpx.Response(200, content=json.dumps({"sequence id": 1})))
    mocker.patch("mx3_beamline_library.plans.basic_scans.SERVER")
    task_info = mocker.patch("mx3_beamline_library.plans.basic_scans.SERVER.retrieveTaskInfo", return_value= [
        "",1,"","","","null",1,
    ])


    user_data = UserData(
        id="my_sample",
        zmq_consumer_mode="spotfinder",
        grid_scan_id="flat",
    )
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
            photon_energy=13
        )
    )

    assert arm.call_count == 1
    assert task_info.call_count==1


@respx.mock(assert_all_mocked=False)
def test_md3_4d_scan(respx_mock, run_engine, mocker: MockerFixture):
    arm = respx_mock.put("http://0.0.0.0:8000/detector/api/1.8.0/command/arm").mock(return_value=httpx.Response(200, content=json.dumps({"sequence id": 1})))
    mocker.patch("mx3_beamline_library.plans.basic_scans.SERVER")
    task_count = mocker.patch("mx3_beamline_library.plans.basic_scans.SERVER.retrieveTaskInfo", return_value= [
        "",1,"","","","null",1,
    ])


    user_data = UserData(
        id="my_sample",
        zmq_consumer_mode="spotfinder",
        grid_scan_id="flat",
    )
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
            photon_energy=13
        )
    )

    assert arm.call_count == 1
    assert task_count.call_count ==1




