from uuid import uuid4

import numpy as np
import pytest
from pytest_mock import MockerFixture

from mx3_beamline_library.plans.crystal_pics import (
    get_grid_scan_crystal_pic,
    get_screen_or_dataset_crystal_pic,
    save_crystal_pic_to_redis,
    save_mxcube_grid_scan_crystal_pic,
)
from mx3_beamline_library.plans.optical_centering import OpticalCentering


def test_save_screen_crystal_pic_to_redis(fake_redis, mocker: MockerFixture):
    # Setup
    mocker.patch("mx3_beamline_library.plans.crystal_pics.redis_connection", fake_redis)
    collection_stage = "start"

    # Execute
    acquisition_uuid = uuid4()
    save_crystal_pic_to_redis(acquisition_uuid, collection_stage)

    # Verify
    key = f"crystal_pic_{collection_stage}:{acquisition_uuid}"  # noqa
    result = fake_redis.get(key)
    assert result is not None
    assert isinstance(result, bytes)


def test_save_dataset_crystal_pic_to_redis(fake_redis, mocker: MockerFixture):
    # Setup
    mocker.patch("mx3_beamline_library.plans.crystal_pics.redis_connection", fake_redis)
    collection_stage = "end"

    # Execute
    acquisition_uuid = uuid4()
    save_crystal_pic_to_redis(acquisition_uuid, collection_stage)

    # Verify
    key = f"crystal_pic_{collection_stage}:{acquisition_uuid}"  # noqa
    result = fake_redis.get(key)
    assert result is not None
    assert isinstance(result, bytes)


def test_get_screen_crystal_pic(fake_redis, mocker: MockerFixture):
    # Setup
    mocker.patch("mx3_beamline_library.plans.crystal_pics.redis_connection", fake_redis)
    collection_stage = "start"
    acquisition_uuid = uuid4()

    save_crystal_pic_to_redis(acquisition_uuid, collection_stage)

    # Execute
    result = get_screen_or_dataset_crystal_pic(acquisition_uuid, collection_stage)

    # Verify
    assert result is not None
    assert isinstance(result, np.ndarray)


def test_get_dataset_crystal_pic(fake_redis, mocker: MockerFixture):
    # Setup
    mocker.patch("mx3_beamline_library.plans.crystal_pics.redis_connection", fake_redis)
    collection_stage = "end"
    acquisition_uuid = uuid4()

    save_crystal_pic_to_redis(acquisition_uuid, collection_stage)

    # Execute
    result = get_screen_or_dataset_crystal_pic(acquisition_uuid, collection_stage)

    # Verify
    assert result is not None
    assert isinstance(result, np.ndarray)


def test_save_mxcube_grid_scan_crystal_pic(fake_redis, mocker: MockerFixture):
    # Setup
    mocker.patch("mx3_beamline_library.plans.crystal_pics.redis_connection", fake_redis)
    acquisition_uuid = uuid4()

    # Execute
    save_mxcube_grid_scan_crystal_pic(acquisition_uuid)

    # Verify
    key = f"mxcube:grid_scan_snapshot:{acquisition_uuid}"  # noqa
    result = fake_redis.get(key)
    assert result is not None
    assert isinstance(result, bytes)


def test_get_grid_scan_mxcube_crystal_pic(fake_redis, mocker: MockerFixture):
    # Setup
    mocker.patch("mx3_beamline_library.plans.crystal_pics.redis_connection", fake_redis)
    acquisition_uuid = uuid4()

    save_mxcube_grid_scan_crystal_pic(acquisition_uuid)

    # Execute
    result = get_grid_scan_crystal_pic(acquisition_uuid=acquisition_uuid)

    # Verify
    assert result is not None
    assert isinstance(result, np.ndarray)


@pytest.mark.parametrize("grid_scan_id", ["flat", "edge"])
def test_get_grid_scan_udc_crystal_pic(
    run_engine, fake_redis, mocker: MockerFixture, sample_id, grid_scan_id
):
    # Setup
    mocker.patch(
        "mx3_beamline_library.plans.optical_centering.redis_connection", fake_redis
    )
    mocker.patch("mx3_beamline_library.plans.crystal_pics.redis_connection", fake_redis)
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
    optical_centering = OpticalCentering(
        sample_id=sample_id,
        beam_position=(612, 512),
        grid_step=(100, 100),
        plot=False,
        calibrated_alignment_z=0.47,
        manual_mode=False,
        use_top_camera_camera=True,
    )

    run_engine(optical_centering.center_loop())

    # Execute
    result = get_grid_scan_crystal_pic(grid_scan_id=grid_scan_id, sample_id=sample_id)

    # Verify
    assert result is not None
    assert isinstance(result, np.ndarray)
