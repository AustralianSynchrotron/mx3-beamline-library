import asyncio
from os import environ

import redis
from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.comm_base import RequestFailedError
from bluesky_queueserver_api.http.aio import REManagerAPI
from prefect import flow, task
import httpx

from mx3_beamline_library.schemas.crystal_finder import MaximumNumberOfSpots, CrystalPositions

AUTHORIZATION_KEY = environ.get("QSERVER_HTTP_SERVER_SINGLE_USER_API_KEY", "666")
BLUESKY_QUEUESERVER_API = environ.get("BLUESKY_QUEUESERVER_API", "http://localhost:60610")
DIALS_API = environ.get("MX_DATA_PROCESSING_DIALS_API", "http://0.0.0.0:8666")

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


@task(name="Mount pin")
async def mount_pin(RM: REManagerAPI, pin_id: int, puck: int) -> None:
    """
    Mounts a pin

    Parameters
    ----------
    RM : REManagerAPI
        Run engine manager
    pin_id : int
        pin id
    puck : int
        Puck

    Returns
    -------
    None
    """
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
        puck=puck,
    )

    await RM.item_add(item)
    await RM.queue_start()
    await RM.wait_for_idle()
    await _check_plan_exit_status(RM)


@task(name="Unmount pin")
async def unmount_pin(RM: REManagerAPI) -> None:
    """
    Unmounts a pin

    Parameters
    ----------
    RM : REManagerAPI
        Run engine manager

    Returns
    -------
    None
    """
    await RM.queue_clear()

    item = BPlan(
        "unmount_pin", unmount_signal="unmount_signal", md3_phase_signal="phase"
    )

    await RM.item_add(item)
    await RM.queue_start()
    await RM.wait_for_idle()
    await _check_plan_exit_status(RM)



@task(name="Optical centering")
async def optical_centering(
    sample_id: str,
    beam_position: tuple[int, int],
    beam_size: tuple[float, float],
) -> None:
    """
    Runs the optical centering task

    Parameters
    ----------
    RM : REManagerAPI
        Run engine manager
    sample_id : str
        Sample id
    beam_position : tuple[int, int]
        beam position
    beam_size : tuple[float, float]
        beam size

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
        beam_size=beam_size,
    )

    await RM.item_add(item)
    await RM.queue_start()
    await RM.wait_for_idle()
    await _check_plan_exit_status(RM)



@task(name="Grid Scan - Flat")
async def grid_scan_flat(
    sample_id: str,
    exposure_time: float,
    omega_range: float = 0,
    count_time: float = None,
    hardware_trigger: bool = True,
) -> None:
    """
    Runs a grid scan - flat

    Parameters
    ----------
    RM : REManagerAPI
        Run engine manager
    sample_id : str
        sample id
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
        "xray_centering",
        sample_id=sample_id,
        detector="dectris_detector",
        omega="omega",
        zoom="zoom",
        exposure_time=exposure_time,
        grid_scan_id="flat",
        omega_range=omega_range,
        count_time=count_time,
        hardware_trigger=hardware_trigger,
    )
    await RM.item_add(item)
    await RM.queue_start()
    await RM.wait_for_idle()
    await _check_plan_exit_status(RM)


@task(name="Grid Scan - Edge")
async def grid_scan_edge(
    sample_id: str,
    exposure_time: float,
    omega_range: float = 0,
    count_time: float = None,
    hardware_trigger: bool = True,
) -> None:
    """
    Runs a grid scan - flat

    Parameters
    ----------
    RM : REManagerAPI
        Run engine manager
    sample_id : str
        sample id
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

    Returns
    -------
    None
    """
    RM = REManagerAPI(http_server_uri=BLUESKY_QUEUESERVER_API)
    RM.set_authorization_key(api_key=AUTHORIZATION_KEY)
    await RM.queue_clear()

    item = BPlan(
        "xray_centering",
        sample_id=sample_id,
        detector="dectris_detector",
        omega="omega",
        zoom="zoom",
        exposure_time=exposure_time,
        grid_scan_id="edge",
        omega_range=omega_range,
        count_time=count_time,
        hardware_trigger=hardware_trigger,
    )
    await RM.item_add(item)
    await RM.queue_start()
    await RM.wait_for_idle()
    await _check_plan_exit_status(RM)

@task(name="Calculate number of spots and find crystals")
async def find_crystals(
    sample_id: str,
    grid_scan_type: str,
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
    data = {"sample_id": sample_id, "grid_scan_type": grid_scan_type, "threshold": threshold}

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{DIALS_API}/v1/spotfinder/find_crystals_in_loop", json=data , timeout=200
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
    sample_id: str,
    crystal_position: CrystalPositions,
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
    crystal_position: CrystalPositions,
        The location of a single crystal in a loop
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
            id=sample_id,
            motor_positions=crystal_position.center_of_mass_motor_coordinates.dict(exclude_none=True),
            number_of_frames=number_of_frames,
            scan_range=scan_range,
            exposure_time=exposure_time,
            number_of_passes=number_of_passes,
            tray_scan=False,
            hardware_trigger=hardware_trigger,
        )
    )

    await RM.queue_start()
    await RM.wait_for_idle()
    await _check_plan_exit_status(RM)

@flow(name="Optical and X ray Centering", retries=1, retry_delay_seconds=1)
async def optical_and_xray_centering(
    sample_id: str,
    pin_id: int,
    puck: int,
    beam_position: tuple[int, int],
    beam_size: tuple[float, float],
    exposure_time: float,
    omega_range: float = 0,
    count_time: float = None,
    scan_number_of_frames: int = 10,
    scan_range: float = 5,
    scan_exposure_time: float = 1,
    scan_number_of_passes: int = 1,
    hardware_trigger: bool = True,
) -> None:
    """
    Runs the optical centering and x ray centering workflow which includes
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

    Returns
    -------
    None
    """
    # _mount = [mount_pin(RM, pin_id=pin_id, puck=puck)]
    await optical_centering(
            sample_id=sample_id, beam_position=beam_position, beam_size=beam_size
    )

    async with asyncio.TaskGroup() as tg:
        _grid_scan_flat = tg.create_task(
            grid_scan_flat(
                sample_id=sample_id,
                exposure_time=exposure_time,
                omega_range=omega_range,
                count_time=count_time,
                hardware_trigger=hardware_trigger
            )
        )
        crystal_finder_results_flat = tg.create_task(find_crystals(sample_id, "flat"))

        while not _grid_scan_flat.done():
            await asyncio.sleep(0.1)

        _grid_scan_edge = tg.create_task(
            grid_scan_edge(
                sample_id=sample_id,
                exposure_time=exposure_time,
                omega_range=omega_range,
                count_time=count_time,
                hardware_trigger=hardware_trigger
            )
        )
        crystal_finder_results_edge = tg.create_task(find_crystals(sample_id, "edge"))

        while not _grid_scan_edge.done():
            await asyncio.sleep(0.1)

        while not crystal_finder_results_flat.done():
            await asyncio.sleep(0.1)
       
        crystals_flat = crystal_finder_results_flat.result()
        
        if crystals_flat is not None:
            crystal_positions_list = crystals_flat[0]
            for crystal in crystal_positions_list:
                await screen_crystal(
                    sample_id=sample_id,
                    crystal_position=crystal,
                    number_of_frames=scan_number_of_frames,
                    scan_range=scan_range,
                    exposure_time=scan_exposure_time,
                    number_of_passes=scan_number_of_passes,
                    hardware_trigger=hardware_trigger,
                )
    

if __name__ == "__main__":
    REDIS_HOST = environ.get("REDIS_HOST", "0.0.0.0")
    REDIS_PORT = int(environ.get("REDIS_PORT", "6379"))
    redis_connection = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    asyncio.run(
        optical_and_xray_centering(
            sample_id="my_sample",
            pin_id=3,
            puck=2,
            beam_position=(640, 512),
            beam_size=(81, 81),
            exposure_time=1,
            hardware_trigger=False,
            scan_range=90,
        )
    )
