import asyncio
from os import environ

from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.comm_base import RequestFailedError
from bluesky_queueserver_api.http.aio import REManagerAPI

from mx3_beamline_library.schemas.detector import UserData

from mx3_beamline_library.science.optical_and_loop_centering.crystal_finder import (
    find_crystals_in_tray
)

from prefect import flow, task
import redis

AUTHORIZATION_KEY = environ.get("QSERVER_HTTP_SERVER_SINGLE_USER_API_KEY", "666")

@task(name="Optical centering")
async def drop_grid_scans(
    http_server_uri: str,
    tray_id: str,
    drop_locations: list[str],
    grid_number_of_columns: int,
    grid_number_of_rows: int,
    exposure_time: float,
    omega_range: float,
    count_time: float | None,
    alignment_y_offset: float,
    alignment_z_offset: float,

) -> None:
    """
    Runs the multiple_drop_grid_scan. This plan runs grid scans for the drops
    specified in the drop_locations list

    Parameters
    ----------
    http_server_uri : str
        Bluesky queueserver http_server_uri
    tray_id: str
        The id of the tray
    drop_location : list[str]
        The drop location, e.g. ["A1-1", "B2-2"]
    grid_number_of_columns : int, optional
        Number of columns of the grid scan, by default 15
    grid_number_of_rows : int, optional
        Number of rows of the grid scan, by default 15
    exposure_time : float, optional
        Exposure time measured in seconds to control shutter command. Note that
        this is the exposure time of one column only, e.g. the md3 takes
        `exposure_time` seconds to move `grid_height` mm.
    omega_range : float, optional
        Omega range of the grid scan, by default 0
    user_data : UserData, optional
        User data pydantic model. This field is passed to the start message
        of the ZMQ stream, by default None
    count_time : float, optional
        Detector count time. If this parameter is not set, it is set to
        frame_time - 0.0000001 by default. This calculation is done via
        the DetectorConfiguration pydantic model.
    alignment_y_offset : float, optional
        Alignment y offset, determined experimentally, by default 0.2
    alignment_z_offset : float, optional
        ALignment z offset, determined experimentally, by default -1.0
    """
    RM = REManagerAPI(http_server_uri=http_server_uri)
    RM.set_authorization_key(api_key=AUTHORIZATION_KEY)

    try:
        await RM.environment_open()
        await RM.wait_for_idle()
    except RequestFailedError:
        print("RM is open")

    await RM.queue_clear()

    item = BPlan(
        "multiple_drop_grid_scan",
        detector="dectris_detector", 
        drop_locations=drop_locations, 
        grid_number_of_columns= grid_number_of_columns,
        grid_number_of_rows=grid_number_of_rows,
        exposure_time=exposure_time,
        omega_range=omega_range,
        user_data=UserData(id=tray_id, zmq_consumer_mode="spotfinder").dict(),
        count_time=count_time,
        alignment_y_offset=alignment_y_offset,
        alignment_z_offset=alignment_z_offset,
    )

    await RM.item_add(item)
    await RM.queue_start()

@task(name="Find crystals")
async def find_crystals(redis_connection, tray_id: str, drop_location: str) -> None:
    """
    Finds crystals after running a grid scan

    Parameters
    ----------
    redis_connection : redis.StrictRedis
        redis connection
    tray_id : str
        The tray_id
    drop_location: str
        The locations of a single drop, e.g. "A1-1"

    Returns
    -------
    None
    """
    await find_crystals_in_tray(
        redis_connection, tray_id=tray_id, drop_location=drop_location
    )



@flow(name="Tray screening")
async def tray_screening_flow(
    http_server_uri: str,
    redis_connection,
    tray_id: str,
    drop_locations: list[str],
    grid_number_of_columns: int = 15,
    grid_number_of_rows: int = 15,
    exposure_time: float = 1,
    omega_range: float = 0,
    count_time: float | None = None,
    alignment_y_offset: float = 0.2,
    alignment_z_offset: float = -1.0,
) -> None:
    """
    Runs the tray screening flow which includes
    1) Mounting a tray
    2) Running grid scans on all drops specified in the drop_locations list
    4) Find the crystals in each drop

    Parameters
    ----------
    http_server_uri : str
        Run engine uri
    redis_connection : redis.StrictRedis
        redis connection
    drop_location : list[str]
        The drop location, e.g. ["A1-1", "B2-2"]
    grid_number_of_columns : int, optional
        Number of columns of the grid scan, by default 15
    grid_number_of_rows : int, optional
        Number of rows of the grid scan, by default 15
    exposure_time : float, optional
        Exposure time measured in seconds to control shutter command. Note that
        this is the exposure time of one column only, e.g. the md3 takes
        `exposure_time` seconds to move `grid_height` mm, by default 1 second
    omega_range : float, optional
        Omega range of the grid scan, by default 0
    count_time : float, optional
        Detector count time. If this parameter is None, it defaults to
        frame_time - 0.0000001. This calculation is done via
        the DetectorConfiguration pydantic model.
    alignment_y_offset : float, optional
        Alignment y offset, determined experimentally, by default 0.2
    alignment_z_offset : float, optional
        Alignment z offset, determined experimentally, by default -1.0

    Returns
    -------
    None
    """
    async_tasks = []

    _drop_grid_scans = drop_grid_scans(
        http_server_uri=http_server_uri,
        tray_id=tray_id,
        drop_locations=drop_locations,
        grid_number_of_columns= grid_number_of_columns,
        grid_number_of_rows=grid_number_of_rows,
        exposure_time=exposure_time,
        omega_range=omega_range,
        count_time=count_time,
        alignment_y_offset=alignment_y_offset,
        alignment_z_offset=alignment_z_offset,
        )
    async_tasks.append(_drop_grid_scans)

    for drop_location in drop_locations:
        async_tasks.append(find_crystals(redis_connection, tray_id, drop_location))


    await asyncio.gather(*async_tasks)


 
if __name__ == "__main__":
    REDIS_HOST = environ.get("REDIS_HOST", "0.0.0.0")
    REDIS_PORT = int(environ.get("REDIS_PORT", "6379"))
    redis_connection = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    asyncio.run(
        tray_screening_flow(
            http_server_uri="http://localhost:60610",
            redis_connection=redis_connection,
            tray_id="my_tray",
            drop_locations=["A1-1", "B1-1", "H8-2"],
            grid_number_of_columns= 10,
            grid_number_of_rows=10,
        )
    )
