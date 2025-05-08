import numpy as np
from pytest_mock import MockerFixture

from mx3_beamline_library.plans.crystal_pics import (
    get_grid_scan_crystal_pic,
    get_screen_or_dataset_crystal_pic,
    save_mxcube_grid_scan_crystal_pic,
    save_screen_or_dataset_crystal_pic_to_redis,
)


def test_save_screen_crystal_pic_to_redis(fake_redis, mocker: MockerFixture):
    # Setup
    mocker.patch("mx3_beamline_library.plans.crystal_pics.redis_connection", fake_redis)
    sample_id = "test_sample"
    crystal_counter = 1
    data_collection_counter = 1
    type = "screening"
    collection_stage = "start"

    # Execute
    save_screen_or_dataset_crystal_pic_to_redis(
        sample_id,
        crystal_counter,
        data_collection_counter,
        type,
        collection_stage,
    )

    # Verify
    key = f"screening_pic:{collection_stage}:sample_{sample_id}:crystal_{crystal_counter}:data_collection_{data_collection_counter}"  # noqa
    result = fake_redis.get(key)
    assert result is not None
    assert isinstance(result, bytes)


def test_save_dataset_crystal_pic_to_redis(fake_redis, mocker: MockerFixture):
    # Setup
    mocker.patch("mx3_beamline_library.plans.crystal_pics.redis_connection", fake_redis)
    sample_id = "test_sample"
    crystal_counter = 1
    data_collection_counter = 1
    type = "dataset"
    collection_stage = "start"

    # Execute
    save_screen_or_dataset_crystal_pic_to_redis(
        sample_id,
        crystal_counter,
        data_collection_counter,
        type,
        collection_stage,
    )

    # Verify
    key = f"dataset_pic:{collection_stage}:sample_{sample_id}:crystal_{crystal_counter}:data_collection_{data_collection_counter}"  # noqa
    result = fake_redis.get(key)
    assert result is not None
    assert isinstance(result, bytes)


def test_get_screen_crystal_pic(fake_redis, mocker: MockerFixture):
    # Setup
    mocker.patch("mx3_beamline_library.plans.crystal_pics.redis_connection", fake_redis)
    sample_id = "test_sample"
    crystal_counter = 1
    data_collection_counter = 1
    type = "screening"
    collection_stage = "start"

    save_screen_or_dataset_crystal_pic_to_redis(
        sample_id,
        crystal_counter,
        data_collection_counter,
        type,
        collection_stage,
    )

    # Execute
    result = get_screen_or_dataset_crystal_pic(
        sample_id,
        crystal_counter,
        data_collection_counter,
        type,
        collection_stage,
    )

    # Verify
    assert result is not None
    assert isinstance(result, np.ndarray)


def test_get_dataset_crystal_pic(fake_redis, mocker: MockerFixture):
    # Setup
    mocker.patch("mx3_beamline_library.plans.crystal_pics.redis_connection", fake_redis)
    sample_id = "test_sample"
    crystal_counter = 1
    data_collection_counter = 1
    type = "dataset"
    collection_stage = "start"

    save_screen_or_dataset_crystal_pic_to_redis(
        sample_id,
        crystal_counter,
        data_collection_counter,
        type,
        collection_stage,
    )

    # Execute
    result = get_screen_or_dataset_crystal_pic(
        sample_id,
        crystal_counter,
        data_collection_counter,
        type,
        collection_stage,
    )

    # Verify
    assert result is not None
    assert isinstance(result, np.ndarray)


def test_save_mxcube_grid_scan_crystal_pic(fake_redis, mocker: MockerFixture):
    # Setup
    mocker.patch("mx3_beamline_library.plans.crystal_pics.redis_connection", fake_redis)
    sample_id = "test_sample"
    grid_scan_id = "test_grid_scan"

    # Execute
    save_mxcube_grid_scan_crystal_pic(
        sample_id,
        grid_scan_id,
    )

    # Verify
    key = f"mxcube_grid_scan_snapshot_{grid_scan_id}:{sample_id}"  # noqa
    result = fake_redis.get(key)
    assert result is not None
    assert isinstance(result, bytes)


def test_get_grid_scan_crystal_pic(fake_redis, mocker: MockerFixture):
    # Setup
    mocker.patch("mx3_beamline_library.plans.crystal_pics.redis_connection", fake_redis)
    sample_id = "test_sample"
    grid_scan_id = "1"

    save_mxcube_grid_scan_crystal_pic(
        sample_id,
        grid_scan_id,
    )

    # Execute
    result = get_grid_scan_crystal_pic(
        sample_id,
        grid_scan_id,
    )

    # Verify
    assert result is not None
    assert isinstance(result, np.ndarray)
