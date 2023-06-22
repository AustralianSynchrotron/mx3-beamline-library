import asyncio
from os import environ

import redis
from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.comm_base import RequestFailedError
from bluesky_queueserver_api.http.aio import REManagerAPI
from prefect import flow, task

from mx3_beamline_library.schemas.crystal_finder import (
    CrystalPositions,
    MaximumNumberOfSpots,
)
from mx3_beamline_library.science.optical_and_loop_centering.crystal_finder import (
    find_crystals_in_tray,
)

AUTHORIZATION_KEY = environ.get("QSERVER_HTTP_SERVER_SINGLE_USER_API_KEY", "666")

REDIS_HOST = environ.get("REDIS_HOST", "0.0.0.0")
REDIS_PORT = int(environ.get("REDIS_PORT", "6379"))
REDIS_CONNECTION = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)


@task(name="Mount tray")
async def mount_tray(http_server_uri: str, tray_location: int) -> None:
    """
    Mounts a tray on the MD3

    Parameters
    ----------
    http_server_uri : str
        Bluesky queueserver http_server_uri
    tray_location: int
        The location of the tray in the tray hotel

    Returns
    -------
    None
    """
    RM = REManagerAPI(http_server_uri=http_server_uri)
    RM.set_authorization_key(api_key=AUTHORIZATION_KEY)

    try:
        await RM.environment_open()
        await RM.wait_for_idle()
    except RequestFailedError:
        print("RM is open")

    await RM.queue_clear()
    await RM.item_add(BPlan("mount_tray", id=tray_location))
    await RM.queue_start()
    await RM.wait_for_idle()


@task(name="Unmount tray")
async def unmount_tray(http_server_uri: str) -> None:
    """
    Unmounts a tray

    Parameters
    ----------
    http_server_uri : str
        Bluesky queueserver http_server_uri

    Returns
    -------
    None
    """
    RM = REManagerAPI(http_server_uri=http_server_uri)
    RM.set_authorization_key(api_key=AUTHORIZATION_KEY)

    await RM.queue_clear()
    await RM.item_add(BPlan("unmount_tray"))
    await RM.queue_start()
    await RM.wait_for_idle()


@task(name="Drop grid scan")
async def drop_grid_scan(
    http_server_uri: str,
    tray_id: str,
    drop_location: str,
    grid_number_of_columns: int,
    grid_number_of_rows: int,
    exposure_time: float,
    omega_range: float,
    count_time: float | None,
    alignment_y_offset: float,
    alignment_z_offset: float,
    hardware_trigger: bool = True,
) -> None:
    """
    Runs a grid scan on a specified drop location.
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
    hardware_trigger : bool, optional
        If set to true, we trigger the detector via hardware trigger, by default True.
        Warning! hardware_trigger=False is used mainly for debugging purposes,
        as it results in a very slow scan

    Returns
    -------
    None
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
        "single_drop_grid_scan",
        tray_id=tray_id,
        detector="dectris_detector",
        drop_location=drop_location,
        grid_number_of_columns=grid_number_of_columns,
        grid_number_of_rows=grid_number_of_rows,
        exposure_time=exposure_time,
        omega_range=omega_range,
        count_time=count_time,
        alignment_y_offset=alignment_y_offset,
        alignment_z_offset=alignment_z_offset,
        hardware_trigger=hardware_trigger,
    )

    await RM.item_add(item)
    await RM.queue_start()
    await RM.wait_for_idle()


@task(name="Find crystals")
async def find_crystals(
    tray_id: str, drop_location: str
) -> tuple[list[CrystalPositions], MaximumNumberOfSpots]:
    """
    Finds crystals after running a grid scan

    Parameters
    ----------
    tray_id : str
        The tray_id
    drop_location: str
        The locations of a single drop, e.g. "A1-1"

    Returns
    -------
    A list of crystal positions
    """
    (
        crystal_locations,
        _,
        maximum_number_of_spots_location,
    ) = await find_crystals_in_tray(
        REDIS_CONNECTION,
        tray_id=tray_id,
        drop_location=drop_location,
        filename=f"crystal_finder_results_{drop_location}",
    )
    return crystal_locations, maximum_number_of_spots_location


@task(name="Screen Crystal")
async def screen_crystal(
    http_server_uri: str,
    maximum_number_of_spots: MaximumNumberOfSpots,
    number_of_frames: int,
    scan_range: float,
    exposure_time: float,
    number_of_passes: int,
    hardware_trigger: bool,
) -> None:
    """
    Screens a crystal

    Parameters
    ----------
    http_server_uri : str
        Bluesky queueserver http_server_uri
    maximum_number_of_spots: MaximumNumberOfSpots
        A MaximumNumberOfSpots pydantic model
    number_of_frames: int
        The detector number of frames
    scan_range : float
        The range of the scan in degrees
    exposure_time : float
        The exposure time in seconds
    number_of_passes : int, optional
        The number of passes, by default 1
    hardware_trigger : bool, optional
        If set to true, we trigger the detector via hardware trigger, by default True.
        Warning! hardware_trigger=False is used mainly for debugging purposes,
        as it results in a very slow scan

    Returns
    -------
    None
    """
    RM = REManagerAPI(http_server_uri=http_server_uri)
    RM.set_authorization_key(api_key=AUTHORIZATION_KEY)

    await RM.queue_clear()

    await RM.item_add(
        BPlan(
            "md3_scan",
            tray_id="my_tray",
            motor_positions=maximum_number_of_spots.motor_positions.dict(),
            scan_idx=0,
            number_of_frames=number_of_frames,
            scan_range=scan_range,
            exposure_time=exposure_time,
            number_of_passes=number_of_passes,
            tray_scan=True,
            hardware_trigger=hardware_trigger,
        )
    )

    await RM.queue_start()
    await RM.wait_for_idle()


async def execute_grid_scans_and_find_crystals(
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
    hardware_trigger: bool = True,
) -> list[tuple[list[CrystalPositions], MaximumNumberOfSpots]]:
    """
    Runs multiple grid scans and finds crystals asynchronously. This function
    runs two prefect tasks: drop_grid_scan and find_crystals.

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
    hardware_trigger : bool, optional
        If set to true, we trigger the detector via hardware trigger, by default True.
        Warning! hardware_trigger=False is used mainly for debugging purposes,
        as it results in a very slow scan


    Returns
    -------
    list[tuple[list[CrystalPositions], MaximumNumberOfSpots]]
        A list of the the crystals locations and the maximum number of spots
        location for each drop
    """

    await drop_grid_scan(
        http_server_uri=http_server_uri,
        tray_id=tray_id,
        drop_location=drop_locations[0],
        grid_number_of_columns=grid_number_of_columns,
        grid_number_of_rows=grid_number_of_rows,
        exposure_time=exposure_time,
        omega_range=omega_range,
        count_time=count_time,
        alignment_y_offset=alignment_y_offset,
        alignment_z_offset=alignment_z_offset,
        hardware_trigger=hardware_trigger,
    )
    crystal_finder_results = []

    if len(drop_locations) > 1:
        for i in range(1, len(drop_locations)):
            drop_grid_scan_async = drop_grid_scan(
                http_server_uri=http_server_uri,
                tray_id=tray_id,
                drop_location=drop_locations[i],
                grid_number_of_columns=grid_number_of_columns,
                grid_number_of_rows=grid_number_of_rows,
                exposure_time=exposure_time,
                omega_range=omega_range,
                count_time=count_time,
                alignment_y_offset=alignment_y_offset,
                alignment_z_offset=alignment_z_offset,
                hardware_trigger=hardware_trigger,
            )
            find_crystals_async = find_crystals(tray_id, drop_locations[i - 1])

            async_result = await asyncio.gather(
                drop_grid_scan_async, find_crystals_async
            )
            crystal_finder_results.append(async_result[1])

    last_drop = await find_crystals(tray_id, drop_locations[-1])
    crystal_finder_results.append(last_drop)

    return crystal_finder_results


@flow(name="Tray screening")
async def tray_screening_flow(
    http_server_uri: str,
    tray_id: str,
    tray_location: int,
    drop_locations: list[str],
    grid_number_of_columns: int = 15,
    grid_number_of_rows: int = 15,
    grid_scan_exposure_time: float = 1,
    grid_scan_omega_range: float = 0,
    grid_scan_count_time: float | None = None,
    alignment_y_offset: float = 0.2,
    alignment_z_offset: float = -1.0,
    scan_number_of_frames: int = 10,
    scan_range: float = 5,
    scan_exposure_time: float = 1,
    scan_number_of_passes: int = 1,
    hardware_trigger: bool = True,
) -> None:
    """
    Runs the tray screening flow which includes
    1) Mounting a tray
    2) Running grid scans on all drops specified in the drop_locations list
    4) Finding crystals
    5) Screening crystals
    6) Unmounting a tray

    Parameters
    ----------
    http_server_uri : str
        Run engine uri
    tray_location: int
        The location of the tray in the tray hotel
    drop_location : list[str]
        The drop location, e.g. ["A1-1", "B2-2"]
    grid_number_of_columns : int, optional
        Number of columns of the grid scan, by default 15
    grid_number_of_rows : int, optional
        Number of rows of the grid scan, by default 15
    grid_scan_exposure_time : float, optional
        Exposure time measured in seconds to control shutter command. Note that
        this is the exposure time of one column only, e.g. the md3 takes
        `exposure_time` seconds to move `grid_height` mm, by default 1 second
    grid_scan_omega_range : float, optional
        Omega range of the grid scan, by default 0
    count_time : float, optional
        Detector count time. If this parameter is None, it defaults to
        frame_time - 0.0000001. This calculation is done via
        the DetectorConfiguration pydantic model.
    alignment_y_offset : float, optional
        Alignment y offset, determined experimentally, by default 0.2
    alignment_z_offset : float, optional
        Alignment z offset, determined experimentally, by default -1.0
    scan_number_of_frames : int
        The number of frames triggered for each scan
    scan_range: float
        The scan range for each scan
    scan_exposure_time: float, optional
        The exposure time of each scan, by default 1
    scan_number_of_passes: int, optional
        The number_of_passes of each scan, by default 1
    hardware_trigger : bool, optional
        If set to true, we trigger the detector via hardware trigger, by default True.
        Warning! hardware_trigger=False is used mainly for debugging purposes,
        as it results in a very slow scan

    Returns
    -------
    None
    """
    drop_locations.sort() # Sort drop locations for efficiency purposes

    await mount_tray(http_server_uri=http_server_uri, tray_location=tray_location)

    crystal_finder_results = await execute_grid_scans_and_find_crystals(
        http_server_uri=http_server_uri,
        tray_id=tray_id,
        drop_locations=drop_locations,
        grid_number_of_columns=grid_number_of_columns,
        grid_number_of_rows=grid_number_of_rows,
        exposure_time=grid_scan_exposure_time,
        omega_range=grid_scan_omega_range,
        count_time=grid_scan_count_time,
        alignment_y_offset=alignment_y_offset,
        alignment_z_offset=alignment_z_offset,
        hardware_trigger=hardware_trigger,
    )

    for results in crystal_finder_results:
        maximum_number_of_spots = results[1]
        if maximum_number_of_spots is not None:
            await screen_crystal(
                http_server_uri=http_server_uri,
                maximum_number_of_spots=maximum_number_of_spots,
                number_of_frames=scan_number_of_frames,
                scan_range=scan_range,
                exposure_time=scan_exposure_time,
                number_of_passes=scan_number_of_passes,
                hardware_trigger=hardware_trigger,
            )

    await unmount_tray(http_server_uri=http_server_uri)


if __name__ == "__main__":
    asyncio.run(
        tray_screening_flow(
            http_server_uri="http://localhost:60610",
            tray_id="my_tray",
            tray_location=1,
            drop_locations=["D7-1"],
            grid_number_of_columns=5,
            grid_number_of_rows=5,
            hardware_trigger=False,
            scan_range=5,
            scan_exposure_time=3,
        )
    )
