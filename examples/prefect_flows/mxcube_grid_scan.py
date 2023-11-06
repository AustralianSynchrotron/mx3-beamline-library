"""
Runs a grid scan in mxcube. While running the scan, the number of
spots are calculated asynchronously.
"""

import asyncio
import logging
from os import environ

import httpx
import redis.asyncio
from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.comm_base import RequestFailedError
from bluesky_queueserver_api.http.aio import REManagerAPI
from prefect import flow, task

from mx3_beamline_library.schemas.crystal_finder import (
    CrystalPositions,
    MaximumNumberOfSpots,
)

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)


AUTHORIZATION_KEY = environ.get("QSERVER_HTTP_SERVER_SINGLE_USER_API_KEY", "666")
BLUESKY_QUEUESERVER_API = environ.get(
    "BLUESKY_QUEUESERVER_API", "http://localhost:60610"
)
DIALS_API = environ.get("MX_DATA_PROCESSING_DIALS_API", "http://0.0.0.0:8666")

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
async def grid_scan(
    sample_id: str,
    grid_scan_id: str | int,
    grid_top_left_coordinate: tuple[int, int],
    grid_height: int,
    grid_width: int,
    beam_position: tuple[int, int] | list[int],
    number_of_columns: int,
    number_of_rows: int,
    exposure_time: float,
    omega_range: float = 0,
    count_time: float = None,
    hardware_trigger: bool = True,
) -> None:
    """
    Runs a grid scan

    Parameters
    ----------
    sample_id : str
        sample id
    grid_scan_id: str | int,
        The id of the grid scan. This id is determined by mxcube
    grid_top_left_coordinate: tuple[int, int]
        Top left coordinates in pixels
    grid_height : int
        The height of the grid in pixels
    grid_width : int
        The width of the grid in pixels
    beam_position : tuple[int, int] | list[int]
        The position of the beam in pixels
    number_of_columns : int
        The number of columns of the grid
    number_of_rows : int
        The number of rows of the grid
    exposure_time : float
        exposure_time : float
        Exposure time measured in seconds to control shutter command. Note that
        this is the exposure time of one column only, e.g. the md3 takes
        `exposure_time` seconds to move `grid_height` mm.
    omega_range : float, optional
        Scan range in degrees, by default 0
    count_time : float, optional
        Detector count time. If this parameter is not set, it is set to
        frame_time - 0.0000001 by default.
    hardware_trigger : bool, optional
        If set to true, we trigger the detector via hardware trigger, by default True.
        Warning! hardware_trigger=False is used mainly for debugging purposes,
        as it results in a very slow scan

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
        "manual_xray_centering",
        sample_id=sample_id,
        detector="dectris_detector",
        omega="omega",
        zoom="zoom",
        grid_top_left_coordinate=grid_top_left_coordinate,
        grid_height=grid_height,
        grid_width=grid_width,
        beam_position=beam_position,
        number_of_columns=number_of_columns,
        number_of_rows=number_of_rows,
        exposure_time=exposure_time,
        grid_scan_id=grid_scan_id,
        omega_range=omega_range,
        count_time=count_time,
        hardware_trigger=hardware_trigger,
    )
    await RM.item_add(item)
    await RM.queue_start()
    await RM.wait_for_idle()
    await _check_plan_exit_status(RM)


@task()
async def find_number_of_spots(
    sample_id: str,
    grid_scan_id: str,
    number_of_rows: int,
    number_of_columns: int,
    threshold: int = 5,
) -> tuple[
    list[CrystalPositions] | None, list[dict] | None, MaximumNumberOfSpots | None
]:
    """
    Finds crystals after running a grid scan

    Parameters
    ----------
    sample_id : str
        The sample id
    grid_scan_id : str
        The grid scan id. It usually comes from mxcube
    drop_location: str
        The locations of a single drop, e.g. "A1-1"

    Returns
    -------
    A list of crystal positions
    """
    # TODO: Remember to change this on loop and tray flows:
    data = {
        "sample_id": sample_id,
        "grid_scan_id": grid_scan_id,
        "grid_scan_type": "loop_mxcube",
        "threshold": threshold,
        "number_of_rows": number_of_rows,
        "number_of_columns": number_of_columns,
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{DIALS_API}/v1/spotfinder/find_crystals_in_loop", json=data, timeout=200
        )

    result = r.json()
    _crystal_locations = result["crystal_locations"]
    _maximum_number_of_spots_location = result["maximum_number_of_spots_location"]
    _distance_between_crystals = result["distance_between_crystals"]

    if _crystal_locations is not None:
        crystal_locations = []
        distance_between_crystals = []
        for crystal in _crystal_locations:
            crystal_locations.append(CrystalPositions.parse_obj(crystal))
            distance_between_crystals.append(_distance_between_crystals)
    else:
        crystal_locations = _crystal_locations
        distance_between_crystals = _distance_between_crystals

    if _maximum_number_of_spots_location is not None:
        maximum_number_of_spots_location = MaximumNumberOfSpots.parse_obj(
            result["maximum_number_of_spots_location"]
        )
    else:
        maximum_number_of_spots_location = _maximum_number_of_spots_location

    return (
        crystal_locations,
        distance_between_crystals,
        maximum_number_of_spots_location,
    )


@flow(retries=1, retry_delay_seconds=1)
async def mxcube_grid_scan(
    sample_id: str,
    grid_scan_id: str | int,
    grid_top_left_coordinate: tuple[int, int],
    grid_height: int,
    grid_width: int,
    number_of_columns: int,
    number_of_rows: int,
    beam_position: tuple[int, int] | list[int] = (640, 512),
    exposure_time: float = 1,
    omega_range: float = 0,
    count_time: float = None,
    hardware_trigger: bool = True,
) -> None:
    """
    Runs a grid scan in mxcube. While running the scan, the number of
    spots are calculated asynchronously.

    Parameters
    ----------
    sample_id : str
        sample id
    grid_scan_id: str | int,
        The id of the grid scan. This id is determined by mxcube
    grid_top_left_coordinate: tuple[int, int]
        Top left coordinates in pixels
    grid_height : int
        The height of the grid in pixels
    grid_width : int
        The width of the grid in pixels
    beam_position : tuple[int, int] | list[int]
        The position of the beam in pixels
    number_of_columns : int
        The number of columns of the grid
    number_of_rows : int
        The number of rows of the grid
    exposure_time : float
        exposure_time : float
        Exposure time measured in seconds to control shutter command. Note that
        this is the exposure time of one column only, e.g. the md3 takes
        `exposure_time` seconds to move `grid_height` mm.
    omega_range : float, optional
        Scan range in degrees, by default 0
    count_time : float, optional
        Detector count time. If this parameter is not set, it is set to
        frame_time - 0.0000001 by default.
    hardware_trigger : bool, optional
        If set to true, we trigger the detector via hardware trigger, by default True.
        Warning! hardware_trigger=False is used mainly for debugging purposes,
        as it results in a very slow scan

    Returns
    -------
    None
    """

    async with asyncio.TaskGroup() as tg:
        tg.create_task(
            grid_scan(
                sample_id=sample_id,
                exposure_time=exposure_time,
                omega_range=omega_range,
                grid_scan_id=grid_scan_id,
                grid_top_left_coordinate=grid_top_left_coordinate,
                grid_height=grid_height,
                grid_width=grid_width,
                beam_position=beam_position,
                number_of_columns=number_of_columns,
                number_of_rows=number_of_rows,
                count_time=count_time,
                hardware_trigger=hardware_trigger,
            )
        )
        tg.create_task(
            find_number_of_spots(
                sample_id=sample_id,
                grid_scan_id=grid_scan_id,
                number_of_columns=number_of_columns,
                number_of_rows=number_of_rows,
            )
        )


if __name__ == "__main__":
    asyncio.run(
        mxcube_grid_scan(
            sample_id="my_sample",
            grid_scan_id="1",
            grid_top_left_coordinate=(512, 600),
            grid_height=100,
            grid_width=100,
            number_of_columns=3,
            number_of_rows=3,
        )
    )
