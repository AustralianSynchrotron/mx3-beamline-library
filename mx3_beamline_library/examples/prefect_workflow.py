from prefect import flow, task
import requests

from os import environ

import requests
from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.comm_base import RequestFailedError
from bluesky_queueserver_api.http.aio import REManagerAPI
import asyncio
from mx3_beamline_library.science.optical_and_loop_centering.crystal_finder import find_crystal_positions
import redis
from typing import Any, Union

AUTHORIZATION_KEY = environ.get(
    "QSERVER_HTTP_SERVER_SINGLE_USER_API_KEY", "666")

from os import environ


@task(name="Mount pin")
async def mount_pin(RM: REManagerAPI, pin_id: int, puck: int):
    await RM.queue_clear()

    try:
        await RM.environment_open()
        await RM.wait_for_idle()
    except RequestFailedError:
        print("Run engine is open, nothing to do here")

    item = BPlan(
        "mount_pin",
        mount_signal="mount_signal",
        md3_phase_signal="phase",
        id=pin_id, 
        puck=puck
    )

    await RM.item_add(item)
    await RM.queue_start()
    await RM.wait_for_idle()

    status = await RM.status()
    print(f"status={status}")

@task(name="Unmount pin")
async def unmount_pin(RM: REManagerAPI):
    await RM.queue_clear()

    item = BPlan(
        "unmount_pin",
        unmount_signal="unmount_signal",
        md3_phase_signal="phase"
    )

    await RM.item_add(item)
    await RM.queue_start()
    await RM.wait_for_idle()

    status = await RM.status()
    print(f"status={status}")


@task(name="Optical centering")
async def optical_centering(
    RM: REManagerAPI, sample_id: str, 
    beam_position: tuple[int,int], beam_size: tuple[float, float]):
    await RM.queue_clear()
    item = BPlan(
        "optical_centering",
        sample_id=sample_id,
        camera="camera",
        sample_x="sample_x",
        sample_y="sample_y",
        alignment_x="alignment_x",
        alignment_y="alignment_y",
        alignment_z="alignment_z",
        omega="omega",
        zoom="zoom",
        phase="phase",
        backlight="backlight",
        beam_position=beam_position,
        beam_size=beam_size,
    )

    await RM.item_add(item)
    await RM.queue_start()
    await RM.wait_for_idle()

    status = await RM.status()
    print(f"status={status}")

@task(name="Grid Scan - Flat")
async def grid_scan_flat(
        RM: REManagerAPI, sample_id: str,exposure_time: float, omega_range: float=0, count_time: float=None
):
    await RM.queue_clear()

    item = BPlan(
        "xray_centering",
        sample_id=sample_id,
        detector="dectris_detector",
        omega="omega",
        zoom="zoom",
        exposure_time=exposure_time,
        grid_scan_type="flat",
        omega_range=omega_range,
        count_time=count_time
    )
    await RM.item_add(item)
    await RM.queue_start()
    await RM.wait_for_idle()

@task(name="Grid Scan - Edge")
async def grid_scan_edge(
    RM: REManagerAPI, sample_id: str,exposure_time: float, omega_range: float=0, count_time: float=None):
    await RM.queue_clear()
    item = BPlan(
        "xray_centering",
        sample_id=sample_id,
        detector="dectris_detector",
        omega="omega",
        zoom="zoom",
        exposure_time=exposure_time,
        grid_scan_type="edge",
        omega_range=omega_range,
        count_time=count_time
    )
    await RM.item_add(item)
    await RM.queue_start()
    await RM.wait_for_idle()

    status = await RM.status()
    print(f"status={status}")

@task(name="Crystal Finder - Edge")
async def find_crystals_edge(redis_connection, sample_id: str):
    """_summary_

    Parameters
    ----------
    redis_connection : redis.StrictRedis
        Redis connection
    sample_id : str
        Sample id
    """
    await find_crystal_positions(
        redis_connection, sample_id=sample_id, grid_scan_type="edge"
        )
    
@task(name="Crystal Finder- Flat")
async def find_crystals_flat(redis_connection, sample_id: str):
    """
    Finds crytals afer the x ray centering step - Flat.

    Parameters
    ----------
    redis_connection : redis.StrictRedis
        redis connection
    sample_id : str
        sample id
    """
    await find_crystal_positions(
        redis_connection, sample_id=sample_id, grid_scan_type="flat")


@flow(name="Optical and X ray Centering",  retries=1, retry_delay_seconds=1)
async def optical_and_xray_centering(
    http_server_uri: str,
    redis_connection,
    sample_id: str,
    pin_id: int, 
    puck: int,
    beam_position: tuple[int, int], 
    beam_size: tuple[float, float], 
    exposure_time: float,
    omega_range: float = 0,
    count_time: float = None
    ):
    """Runs the optical centering plan which includes
    1) Sample Mounting
    2) Optical Centering
    3) X ray centering
    4) Post processing

    Parameters
    ----------
    http_server_uri : str
        Run engine uri
    redis_connection : redis.StrictRedis
        redis connection
    sample_id : str
        Sample id
    pin_id : int
        Pin id
    puck : int
        Puck
    beam_position : tuple[int, int]
        Beam position
    beam_size : tuple[float, float]
        Beam size
    exposure_time : float
        Exposure time measured in seconds to control shutter command. Note that
        this is the exposure time of one column only, e.g. the md3 takes
        `exposure_time` seconds to move `grid_height` mm.
    omega_range : float, optional
        Scan range in degrees, by default 0
    count_time : float, optional
        Detector count time. If this parameter is not set, it is set to
        frame_time - 0.0000001 by default.
    """
    RM = REManagerAPI(http_server_uri=http_server_uri)
    RM.set_authorization_key(api_key=AUTHORIZATION_KEY)
    try:
        await RM.environment_open()
        await RM.wait_for_idle()
    except RequestFailedError:
        print("Run engine is open, nothing to do here")

    _mount = [mount_pin(RM, pin_id=pin_id, puck=puck)]

    _optical_centering = [optical_centering(
        RM, sample_id=sample_id, beam_position=beam_position, beam_size=beam_size)]
 
    _grid_scan_flat = [grid_scan_flat(
        RM, sample_id=sample_id,
        exposure_time=exposure_time,
        omega_range=omega_range, count_time=count_time

        )]

    _spotfinding_and_grid_scan_edge = [
        find_crystals_flat(redis_connection, sample_id=sample_id),
        grid_scan_edge(RM, sample_id=sample_id,
                exposure_time=exposure_time, omega_range=omega_range, count_time=count_time), 
        find_crystals_edge(redis_connection, sample_id=sample_id)
    ]
    _unmount = [unmount_pin(RM)]


    #await asyncio.gather(*_mount)
    await asyncio.gather(*_optical_centering)
    await asyncio.gather(*_grid_scan_flat)
    await asyncio.gather(*_spotfinding_and_grid_scan_edge)
    #await asyncio.gather(*_unmount)

if __name__ == "__main__":
    REDIS_HOST = environ.get("REDIS_HOST", "0.0.0.0")
    REDIS_PORT = int(environ.get("REDIS_PORT", "6379"))
    redis_connection = redis.StrictRedis(
        host=REDIS_HOST, port=REDIS_PORT, db=0
    )
    asyncio.run(
        optical_and_xray_centering(
            http_server_uri="http://localhost:60610",
            redis_connection=redis_connection,
            sample_id="my_test_sample",
            pin_id=3,
            puck=2,
            beam_position=[640, 512],
            beam_size=[80,80],
            exposure_time=0.5
        )
    )