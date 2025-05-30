import json

import httpx
import respx
from pytest_mock.plugin import MockerFixture

from mx3_beamline_library.plans.beam_utils import (
    get_beam_center_4M,
    get_beam_center_16M,
    set_beam_center,
)


def test_get_beam_center_16M(fake_redis, mocker: MockerFixture):
    # Setup
    mocker.patch(
        "mx3_beamline_library.plans.beam_utils.redis_connection", new=fake_redis
    )
    distance = 500.0  # Example distance in mm
    fake_redis.hset(name="beam_center_x_16M", mapping={"a": 2000.0, "b": 1.0, "c": 2.0})
    fake_redis.hset(name="beam_center_y_16M", mapping={"a": 2002.0, "b": 3.0, "c": 4.0})
    # Exercise
    beam_center_x, beam_center_y = get_beam_center_16M(distance)
    # Verify
    assert beam_center_x == 2000.0 + 1.0 * distance + 2.0 * distance**2
    assert beam_center_y == 2002.0 + 3.0 * distance + 4.0 * distance**2


def test_get_beam_center_4M(fake_redis, mocker: MockerFixture):
    # Setup
    mocker.patch(
        "mx3_beamline_library.plans.beam_utils.redis_connection", new=fake_redis
    )
    distance = 500.0  # Example distance in mm
    fake_redis.hset(name="beam_center_x_4M", mapping={"a": 1000.0, "b": 5.0, "c": 6.0})
    fake_redis.hset(name="beam_center_y_4M", mapping={"a": 1007.0, "b": 8.0, "c": 9.0})
    # Exercise
    beam_center_x, beam_center_y = get_beam_center_4M(distance)
    # Verify
    assert beam_center_x == 1000.0 + 5.0 * distance + 6.0 * distance**2
    assert beam_center_y == 1007.0 + 8.0 * distance + 9.0 * distance**2


@respx.mock()
def test_set_beam_center_16M(respx_mock, fake_redis, mocker: MockerFixture):
    # Setup
    roi_mode = respx_mock.get("/detector/api/1.8.0/config/roi_mode").mock(
        return_value=httpx.Response(200, content=json.dumps({"value": "disabled"}))
    )
    beam_center_x_put = respx_mock.put("/detector/api/1.8.0/config/beam_center_x").mock(
        return_value=httpx.Response(200, content=json.dumps({"value": "disabled"}))
    )
    beam_center_y_put = respx_mock.put("/detector/api/1.8.0/config/beam_center_y").mock(
        return_value=httpx.Response(200, content=json.dumps({"value": "disabled"}))
    )
    mocker.patch(
        "mx3_beamline_library.plans.beam_utils.redis_connection", new=fake_redis
    )
    distance = 500.0  # Example distance in mm
    fake_redis.hset(name="beam_center_x_16M", mapping={"a": 2000.0, "b": 1.0, "c": 2.0})
    fake_redis.hset(name="beam_center_y_16M", mapping={"a": 2002.0, "b": 3.0, "c": 4.0})
    # Exercise
    beam_center_x, beam_center_y = set_beam_center(distance)
    # Verify
    assert beam_center_x == 2000.0 + 1.0 * distance + 2.0 * distance**2
    assert beam_center_y == 2002.0 + 3.0 * distance + 4.0 * distance**2
    roi_mode.call_count == 1
    beam_center_x_put.call_count == 1
    beam_center_y_put.call_count == 1


@respx.mock()
def test_set_beam_center_4M(respx_mock, fake_redis, mocker: MockerFixture):
    # Setup
    roi_mode = respx_mock.get("/detector/api/1.8.0/config/roi_mode").mock(
        return_value=httpx.Response(200, content=json.dumps({"value": "4M"}))
    )
    beam_center_x_put = respx_mock.put("/detector/api/1.8.0/config/beam_center_x").mock(
        return_value=httpx.Response(200, content=json.dumps({}))
    )
    beam_center_y_put = respx_mock.put("/detector/api/1.8.0/config/beam_center_y").mock(
        return_value=httpx.Response(200, content=json.dumps({}))
    )
    mocker.patch(
        "mx3_beamline_library.plans.beam_utils.redis_connection", new=fake_redis
    )
    distance = 500.0  # Example distance in mm
    fake_redis.hset(name="beam_center_x_4M", mapping={"a": 1000.0, "b": 5.0, "c": 6.0})
    fake_redis.hset(name="beam_center_y_4M", mapping={"a": 1007.0, "b": 8.0, "c": 9.0})
    # Exercise
    beam_center_x, beam_center_y = set_beam_center(distance)
    # Verify
    assert beam_center_x == 1000.0 + 5.0 * distance + 6.0 * distance**2
    assert beam_center_y == 1007.0 + 8.0 * distance + 9.0 * distance**2
    roi_mode.call_count == 1
    beam_center_x_put.call_count == 1
    beam_center_y_put.call_count == 1
