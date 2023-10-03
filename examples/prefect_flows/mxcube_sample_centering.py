"""
Runs a single_loop_data_collection prefect workflow. This workflow encompasses the following
steps:

    1) Mounting a sample onto the goniometer.
    2) Aligning the tip of the loop with the center of the beam.
    3) Conducting two grid scans to locate crystals within the loop.
    4) Determining the 3D position of the identified crystals.
    5) Screening the center of mass for all crystals found within the loop.
"""

import asyncio
import logging
import pickle
from os import environ
from typing import Union

import httpx
import redis.asyncio
from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.comm_base import RequestFailedError
from bluesky_queueserver_api.http.aio import REManagerAPI
from prefect import flow, task

from mx3_beamline_library.schemas.crystal_finder import (
    CrystalPositions,
    CrystalVolume,
    MaximumNumberOfSpots,
    MotorCoordinates,
)

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)


AUTHORIZATION_KEY = environ.get("QSERVER_HTTP_SERVER_SINGLE_USER_API_KEY", "666")
BLUESKY_QUEUESERVER_API = environ.get(
    "BLUESKY_QUEUESERVER_API", "http://localhost:60610"
)

REDIS_HOST = environ.get("REDIS_HOST", "0.0.0.0")
REDIS_PORT = int(environ.get("REDIS_PORT", "6379"))
REDIS_CONNECTION = redis.asyncio.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)



async def _check_plan_exit_status(RM: REManagerAPI):
    """
    Checks if a bluesky plan has run successfully

    Parameters
    ----------
    RM : REManagerAPI
        Bluesky run engine manager

    Raises
    ------
    RuntimeError
        An error if the bluesky plan has no run successfully
    """
    history = await RM.history_get()
    latest_result = history["items"][-1]["result"]
    exit_status: str = latest_result["exit_status"]

    if exit_status.lower() == "failed":
        raise RuntimeError(f"Plan failed: {latest_result}")
    
@task()
async def _mxcube_sample_centering(
    sample_id: str,
    beam_position: tuple[int, int],
) -> None:
    """
    Runs the optical centering task. This includes zoom level-zero centering, aligning the
    loop with the center of the beam, inferring the flat and edge angles, and determining
    the raster grid coordinates for the flat and edge grid scans

    Parameters
    ----------
    RM : REManagerAPI
        Run engine manager
    sample_id : str
        Sample id
    beam_position : tuple[int, int]
        The (x,y) beam position in pixels
    grid_step : tuple[float, float]
        The grid step in micrometers along the (x,y) axis (in that order)

    Returns
    -------
    None
    """

    RM = REManagerAPI(http_server_uri=BLUESKY_QUEUESERVER_API)
    RM.set_authorization_key(api_key=AUTHORIZATION_KEY)

    try:
        await RM.environment_open()
        await RM.wait_for_idle()
    except RequestFailedError:
        logger.info("RM is open")

    await RM.queue_clear()
    item = BPlan(
        "optical_centering",
        sample_id=sample_id,
        md3_camera="md_camera",
        top_camera="blackfly_camera",
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
        manual_mode=True
    )

    await RM.item_add(item)
    await RM.queue_start()
    await RM.wait_for_idle()
    await _check_plan_exit_status(RM)

@flow()
async def mxcube_sample_centering(
    sample_id: str,
    beam_position: tuple[int, int],
    ):
    await _mxcube_sample_centering(sample_id=sample_id, beam_position=beam_position)

if __name__ == "__main__":
    asyncio.run(
        mxcube_sample_centering(
            sample_id="my_sample",
            beam_position=(640, 512),
        )
    )

