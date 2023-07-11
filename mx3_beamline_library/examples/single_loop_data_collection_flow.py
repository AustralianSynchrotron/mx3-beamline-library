import asyncio
import logging
import pickle
from os import environ

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


async def _check_if_optical_centering_was_successful(sample_id):
    redis_key = await REDIS_CONNECTION.get(f"optical_centering_results:{sample_id}")
    optical_centering_results = pickle.loads(redis_key)
    if not optical_centering_results["optical_centering_successful"]:
        raise ValueError("Optical centering was not successful")
    else:
        logger.info("Optical centering was successful")


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
async def mount_pin(pin_id: int, puck: int) -> None:
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

    RM = REManagerAPI(http_server_uri=BLUESKY_QUEUESERVER_API)
    RM.set_authorization_key(api_key=AUTHORIZATION_KEY)

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
async def unmount_pin() -> None:
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
    RM = REManagerAPI(http_server_uri=BLUESKY_QUEUESERVER_API)
    RM.set_authorization_key(api_key=AUTHORIZATION_KEY)
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
    grid_step: tuple[float, float],
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
    grid_step : tuple[float, float]
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
        grid_step=grid_step,
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
) -> tuple[
    list[CrystalPositions] | None, list[dict] | None, MaximumNumberOfSpots | None
]:
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
    data = {
        "sample_id": sample_id,
        "grid_scan_type": grid_scan_type,
        "threshold": threshold,
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


@task(name="Find crystals in 3D")
async def find_crystals_in_3D(
    flat_coordinates: list[CrystalPositions],
    edge_coordinates: list[CrystalPositions],
    distance_flat: list[dict],
    distance_edge: list[dict],
) -> list[CrystalVolume]:
    """
    Finds crystals in 3D from the edge and flat grid scans

    Parameters
    ----------
    flat_coordinates : list[CrystalPositions]
        A list of flat coordinated obtained from the CrystalFinder
    edge_coordinates : list[CrystalPositions]
        A list of edge coordinated obtained from the CrystalFinder
    distance_flat : list[dict]
        Distance between crystals (flat grid scan)
    distance_edge : list[dict]
        Distance between crystals (edge grid scan)

    Returns
    -------
    list[CrystalVolume]
        A list of CrystalVolume pydantic models
    """

    flat_coords_dict = []
    edge_coords_dict = []
    for i in range(len(flat_coordinates)):
        flat_coords_dict.append(flat_coordinates[i].dict())
        edge_coords_dict.append(edge_coordinates[i].dict())

    data = {
        "flat_coordinates": flat_coords_dict,
        "edge_coordinates": edge_coords_dict,
        "distance_flat": distance_flat,
        "distance_edge": distance_edge,
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{DIALS_API}/v1/spotfinder/find_crystals_in_3D", json=data, timeout=200
        )

    results = r.json()

    crystal_volumes_list = []
    for i in range(len(flat_coordinates)):
        crystal_volumes_list.append(
            CrystalVolume.parse_obj(results["crystal_volumes"][i])
        )

    logger.info(f"3D crystal results: {crystal_volumes_list}")
    return crystal_volumes_list


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
            motor_positions=crystal_position.center_of_mass_motor_coordinates.dict(
                exclude_none=True
            ),
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
    grid_step: tuple[float, float],
    exposure_time: float,
    grid_scan_omega_range: float = 0,
    count_time: float = None,
    scan_number_of_frames: int = 10,
    scan_range: float = 5,
    scan_exposure_time: float = 1,
    scan_number_of_passes: int = 1,
    mount_pin_at_start_of_flow: bool = False,
    unmount_pin_when_flow_ends: bool = False,
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
    sample_id : str
        Sample id
    pin_id : int
        Pin id
    puck : int
        Puck
    beam_position : tuple[int, int]
        Beam position
    grid_step : tuple[float, float]
        Beam size
    exposure_time : float
        Exposure time
    grid_scan_omega_range : float, optional
        Omega range of the grid scan
    count_time : float, optional
        Detector count time. If this parameter is not set, it is set to
        frame_time - 0.0000001 by default.
    mount_pin_at_start_of_flow : bool
        Mounts a pin at the start of the flow, by default False
    unmount_pin_when_flow_ends : bool
        Unmounts a pin at the end of the flow, by default False

    Returns
    -------
    None
    """
    if mount_pin_at_start_of_flow:
        await mount_pin(pin_id=pin_id, puck=puck)

    # await optical_centering(
    #        sample_id=sample_id, beam_position=beam_position, grid_step=grid_step
    # )
    await _check_if_optical_centering_was_successful(sample_id=sample_id)

    async with asyncio.TaskGroup() as tg:
        _grid_scan_flat = tg.create_task(
            grid_scan_flat(
                sample_id=sample_id,
                exposure_time=exposure_time,
                omega_range=grid_scan_omega_range,
                count_time=count_time,
                hardware_trigger=hardware_trigger,
            )
        )
        crystal_finder_results_flat = tg.create_task(find_crystals(sample_id, "flat"))

        while not _grid_scan_flat.done():
            await asyncio.sleep(0.1)

        _grid_scan_edge = tg.create_task(
            grid_scan_edge(
                sample_id=sample_id,
                exposure_time=exposure_time,
                omega_range=grid_scan_omega_range,
                count_time=count_time,
                hardware_trigger=hardware_trigger,
            )
        )
        crystal_finder_results_edge = tg.create_task(find_crystals(sample_id, "edge"))

    crystals_flat = crystal_finder_results_flat.result()
    crystals_edge = crystal_finder_results_edge.result()

    if crystals_flat[0] is not None or crystals_edge[0] is not None:
        crystal_3d_results = await find_crystals_in_3D(
            flat_coordinates=crystals_flat[0],
            edge_coordinates=crystals_edge[0],
            distance_flat=crystals_flat[1],
            distance_edge=crystals_edge[1],
        )

    if crystals_flat[0] is not None:
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

    if unmount_pin_when_flow_ends:
        await unmount_pin()


if __name__ == "__main__":
    asyncio.run(
        optical_and_xray_centering(
            sample_id="my_sample",
            pin_id=3,
            puck=2,
            beam_position=(640, 512),
            grid_step=(81, 81),
            exposure_time=0.02,
            hardware_trigger=True,
            scan_range=90,
        )
    )
