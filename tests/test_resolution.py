import pytest

from mx3_beamline_library.science.resolution import calculate_resolution


def test_resolution_16M():
    resolution = calculate_resolution(
        distance=380,
        energy=13,
        roi_mode="16M",
        beam_center=(2032.2375434478047, 2153.7649990331934),
    )
    expected_res = 2.515458709328944
    assert expected_res == pytest.approx(resolution, rel=1e-6)


def test_resolution_4M():
    resolution = calculate_resolution(
        distance=380,
        energy=13,
        roi_mode="4M",
        beam_center=(992.2375434478047, 1053.7649990331934),
    )
    expected_res = 4.939335003922882
    assert expected_res == pytest.approx(resolution, rel=1e-6)


def test_resolution_16M_redis_beam_center(fake_redis, mocker):
    mocker.patch(
        "mx3_beamline_library.plans.beam_utils.redis_connection", new=fake_redis
    )

    fake_redis.hset(
        name="beam_center_x_16M", mapping={"a": 2032.2375434478047, "b": 0, "c": 0}
    )
    fake_redis.hset(
        name="beam_center_y_16M", mapping={"a": 2153.7649990331934, "b": 0, "c": 0}
    )

    # Exercise
    resolution = calculate_resolution(distance=380, energy=13, roi_mode="16M")
    expected_res = 2.515458709328944
    assert expected_res == pytest.approx(resolution, rel=1e-6)


def test_resolution_4M_redis_beam_center(fake_redis, mocker):
    mocker.patch(
        "mx3_beamline_library.plans.beam_utils.redis_connection", new=fake_redis
    )

    fake_redis.hset(
        name="beam_center_x_4M", mapping={"a": 992.2375434478047, "b": 0, "c": 0}
    )
    fake_redis.hset(
        name="beam_center_y_4M", mapping={"a": 1053.7649990331934, "b": 0, "c": 0}
    )

    # Exercise
    resolution = calculate_resolution(distance=380, energy=13, roi_mode="4M")
    expected_res = 4.939335003922882
    assert expected_res == pytest.approx(resolution, rel=1e-6)
