"""
Runs the tray screening prefect flow which has the following steps:
    1) Mounts a tray
    2) Runs grid scans on all drops specified in the drop_locations list
    4) Finds crystals
    5) Screens one crystal per drop (the crystal with the maximum number of spots)
    6) Unmounts a tray
"""

import asyncio
from os import environ

import httpx
from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.comm_base import RequestFailedError
from bluesky_queueserver_api.http.aio import REManagerAPI
from prefect import flow, task

from mx3_beamline_library.schemas.crystal_finder import (
    CrystalPositions,
    MaximumNumberOfSpots,
)

AUTHORIZATION_KEY = environ.get("QSERVER_HTTP_SERVER_SINGLE_USER_API_KEY", "666")
DIALS_API = environ.get("MX_DATA_PROCESSING_DIALS_API", "http://0.0.0.0:8666")
BLUESKY_QUEUESERVER_API = environ.get("BLUESKY_QUEUESERVER_API", "http://0.0.0.0:8080")


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


@task(name="Mount tray")
async def mount_tray(tray_location: int) -> None:
    """
    Mounts a tray on the MD3

    Parameters
    ----------
    tray_location: int
        The location of the tray in the tray hotel

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
        print("RM is open")

    await RM.queue_clear()
    await RM.item_add(BPlan("mount_tray", id=tray_location))
    await RM.queue_start()
    await RM.wait_for_idle()

    await _check_plan_exit_status(RM)


@task(name="Unmount tray")
async def unmount_tray() -> None:
    """
    Unmounts a tray

    Returns
    -------
    None
    """
    RM = REManagerAPI(http_server_uri=BLUESKY_QUEUESERVER_API)
    RM.set_authorization_key(api_key=AUTHORIZATION_KEY)

    await RM.queue_clear()
    await RM.item_add(BPlan("unmount_tray"))
    await RM.queue_start()
    await RM.wait_for_idle()
    await _check_plan_exit_status(RM)


@task(name="Drop grid scan")
async def drop_grid_scan(
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
    RM = REManagerAPI(http_server_uri=BLUESKY_QUEUESERVER_API)
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
    await _check_plan_exit_status(RM)


@task(name="Calculate number of spots and find crystals")
async def find_crystals(
    tray_id: str,
    drop_location: str,
    threshold: int = 5,
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
    data = {"tray_id": tray_id, "drop_location": drop_location, "threshold": threshold}

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{DIALS_API}/v1/spotfinder/find_crystals_in_tray", json=data, timeout=200
        )

    result = r.json()
    _crystal_locations = result["crystal_locations"]
    _maximum_number_of_spots_location = result["maximum_number_of_spots_location"]

    if _crystal_locations is not None:
        crystal_locations = []
        for crystal in _crystal_locations:
            crystal_locations.append(CrystalPositions.parse_obj(crystal))
    else:
        crystal_locations = _crystal_locations

    if _maximum_number_of_spots_location is not None:
        maximum_number_of_spots_location = MaximumNumberOfSpots.parse_obj(
            result["maximum_number_of_spots_location"]
        )
    else:
        maximum_number_of_spots_location = _maximum_number_of_spots_location

    return crystal_locations, maximum_number_of_spots_location


@task(name="Screen Crystal")
async def screen_crystal(
    tray_id: str,
    maximum_number_of_spots: MaximumNumberOfSpots,
    number_of_frames: int,
    scan_range: float,
    exposure_time: float,
    number_of_passes: int,
    drop_location: str,
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
    RM = REManagerAPI(http_server_uri=BLUESKY_QUEUESERVER_API)
    RM.set_authorization_key(api_key=AUTHORIZATION_KEY)

    await RM.queue_clear()

    await RM.item_add(
        BPlan(
            "md3_scan",
            id=tray_id,
            motor_positions=maximum_number_of_spots.motor_positions.dict(),
            number_of_frames=number_of_frames,
            scan_range=scan_range,
            exposure_time=exposure_time,
            number_of_passes=number_of_passes,
            tray_scan=True,
            drop_location=drop_location,
            hardware_trigger=hardware_trigger,
        )
    )

    await RM.queue_start()
    await RM.wait_for_idle()
    await _check_plan_exit_status(RM)


@flow(name="Tray screening")
async def tray_screening_flow(
    tray_id: str,
    tray_location: int,
    drop_locations: list[str],
    grid_number_of_columns: int = 91,
    grid_number_of_rows: int = 91,
    grid_scan_exposure_time: float = 0.182,
    grid_scan_omega_range: float = 0,
    grid_scan_count_time: float | None = None,
    alignment_y_offset: float = 0.3,
    alignment_z_offset: float = -1.0,
    scan_number_of_frames: int = 10,
    scan_range: float = 5,
    scan_exposure_time: float = 1,
    scan_number_of_passes: int = 1,
    mount_tray_at_start_of_flow: bool = False,
    unmount_tray_when_flow_ends: bool = False,
    hardware_trigger: bool = True,
) -> None:
    """
    Runs the tray screening flow which has the following steps:
    1) Mounts a tray
    2) Runs grid scans on all drops specified in the drop_locations list
    4) Finds crystals
    5) Screens one crystal per drop (the crystal with the maximum number of spots)
    6) Unmounts a tray

    Parameters
    ----------
    tray_location: int
        The location of the tray in the tray hotel
    drop_location : list[str]
        The drop location, e.g. ["A1-1", "B2-2"]
    grid_number_of_columns : int, optional
        Number of columns of the grid scan, by default 91
    grid_number_of_rows : int, optional
        Number of rows of the grid scan, by default 91
    grid_scan_exposure_time : float, optional
        Exposure time measured in seconds to control shutter command. Note that
        this is the exposure time of one column only, e.g. the md3 takes
        `exposure_time` seconds to move `grid_height` mm, by default 0.182
        seconds
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
    mount_tray_at_start_of_flow : bool
        Mounts the tray at the start of the flow, by default False
    unmount_tray_when_flow_ends : bool
        Unmounts the tray at the end of the flow, by default False
    hardware_trigger : bool, optional
        If set to true, we trigger the detector via hardware trigger, by default True.
        Warning! hardware_trigger=False is used mainly for debugging purposes,
        as it results in a very slow scan

    Returns
    -------
    None
    """
    # drop_locations.sort()  # Sort drop locations for efficiency purposes
    if mount_tray_at_start_of_flow:
        await mount_tray(tray_location=tray_location)

    async with asyncio.TaskGroup() as tg:
        crystal_finder_results: list[asyncio.Task] = []
        for drop in drop_locations:
            grid_scan = tg.create_task(
                drop_grid_scan(
                    tray_id=tray_id,
                    drop_location=drop,
                    grid_number_of_columns=grid_number_of_columns,
                    grid_number_of_rows=grid_number_of_rows,
                    exposure_time=grid_scan_exposure_time,
                    omega_range=grid_scan_omega_range,
                    count_time=grid_scan_count_time,
                    alignment_y_offset=alignment_y_offset,
                    alignment_z_offset=alignment_z_offset,
                    hardware_trigger=hardware_trigger,
                )
            )
            crystal_finder = tg.create_task(find_crystals(tray_id, drop))
            crystal_finder_results.append(crystal_finder)

            while not grid_scan.done():
                await asyncio.sleep(0.1)

        for i, result in enumerate(crystal_finder_results):
            while not result.done():
                await asyncio.sleep(0.1)

            maximum_number_of_spots = result.result()[1]

            if maximum_number_of_spots is not None:
                await screen_crystal(
                    tray_id=tray_id,
                    drop_location=drop_locations[i],
                    maximum_number_of_spots=maximum_number_of_spots,
                    number_of_frames=scan_number_of_frames,
                    scan_range=scan_range,
                    exposure_time=scan_exposure_time,
                    number_of_passes=scan_number_of_passes,
                    hardware_trigger=hardware_trigger,
                )

    if unmount_tray_when_flow_ends:
        await unmount_tray()


if __name__ == "__main__":
    grid_number_of_columns = 5
    grid_number_of_rows = 300
    grid_scan_exposure_time = 0.6
    frame_rate = grid_number_of_rows / grid_scan_exposure_time
    print("frame rate", frame_rate)
    asyncio.run(
        tray_screening_flow(
            tray_id="my_tray",
            tray_location=1,
            drop_locations=["D7-1"],
            grid_number_of_columns=grid_number_of_columns,
            grid_number_of_rows=grid_number_of_rows,
            grid_scan_exposure_time=grid_scan_exposure_time,
            hardware_trigger=True,
            scan_range=5,
            scan_exposure_time=3,
        )
    )
