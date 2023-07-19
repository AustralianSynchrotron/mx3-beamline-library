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
        logger.info("Run engine is open, nothing to do here")

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
        grid_step=grid_step,
    )

    await RM.item_add(item)
    await RM.queue_start()
    await RM.wait_for_idle()
    await _check_plan_exit_status(RM)
    await _check_if_optical_centering_was_successful(sample_id=sample_id)


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
        logger.info("RM is open")

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
    sample_id: str,
    beam_position: tuple[int, int] | list,
) -> tuple[list[CrystalVolume], list[Union[MotorCoordinates, None]]]:
    """
    Finds crystals in 3D from the edge and flat grid scans. The values returned are the
    volumes of all crystals found in the loop and the positions of the centers of mass
    of all crystals

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
    tuple[list[CrystalVolume], list[MotorCoordinates]]
        A list of CrystalVolume pydantic models, and list of center of mass motor positions
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
        "sample_id": sample_id,
        "beam_position": list(beam_position),
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

    centers_of_mass_list = []
    for i in range(len(flat_coordinates)):
        centers_of_mass_list.append(
            MotorCoordinates.parse_obj(results["centers_of_mass_positions"][i])
        )

    logger.info(f"3D crystal results: {crystal_volumes_list}")
    return crystal_volumes_list, centers_of_mass_list


@task(name="Screen Crystal")
async def screen_crystal(
    sample_id: str,
    crystal_position: MotorCoordinates,
    start_omega: float,
    number_of_frames: int,
    scan_range: float,
    exposure_time: float,
    number_of_passes: int,
    count_time: float,
    hardware_trigger: bool,
) -> None:
    """
    Screens a crystal

    Parameters
    ----------
    http_server_uri : str
        Bluesky queueserver http_server_uri
    crystal_position: MotorCoordinates,
        The location of the center of mass of a single crystal
    start_omega : float
        The initial screening angle in degrees
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
    crystal_position.omega = start_omega

    RM = REManagerAPI(http_server_uri=BLUESKY_QUEUESERVER_API)
    RM.set_authorization_key(api_key=AUTHORIZATION_KEY)

    await RM.queue_clear()

    await RM.item_add(
        BPlan(
            "md3_scan",
            id=sample_id,
            motor_positions=crystal_position.dict(exclude_none=True),
            number_of_frames=number_of_frames,
            scan_range=scan_range,
            exposure_time=exposure_time,
            number_of_passes=number_of_passes,
            tray_scan=False,
            count_time=count_time,
            hardware_trigger=hardware_trigger,
        )
    )

    await RM.queue_start()
    await RM.wait_for_idle()
    await _check_plan_exit_status(RM)


@flow(name="single_loop_data_collection", retries=1, retry_delay_seconds=1)
async def single_loop_data_collection(
    sample_id: str,
    pin_id: int,
    puck: int,
    beam_position: tuple[int, int],
    grid_step: tuple[float, float],
    grid_scan_exposure_time: float,
    grid_scan_omega_range: float = 0,
    grid_scan_count_time: float = None,
    screening_number_of_frames: int = 10,
    screening_start_omega: float = 0,
    screening_omega_range: float = 5,
    screening_exposure_time: float = 1,
    screening_number_of_passes: int = 1,
    screening_count_time: float = None,
    mount_pin_at_start_of_flow: bool = False,
    unmount_pin_when_flow_ends: bool = False,
    hardware_trigger: bool = True,
) -> None:
    """
    The "single_loop_data_collection" workflow encompasses the following steps:

    1) Mounting a sample onto the goniometer.
    2) Aligning the tip of the loop with the center of the beam.
    3) Conducting two grid scans to locate crystals within the loop.
    4) Determining the 3D position of the identified crystals.
    5) Screening the center of mass for all crystals found within the loop.

    Parameters
    ----------
    sample_id : str
        Sample id
    pin_id : int
        Pin id (used for mounting and unmounting samples)
    puck : int
        Puck (used for mounting and unmounting samples)
    beam_position : tuple[int, int]
        The (x,y) beam position in pixels
    grid_step : tuple[float, float]
        The grid step in micrometers along the (x,y) axis (in that order).
    grid_scan_exposure_time : float
        Exposure time of the grid scans. NOTE: This is NOT the md3 definition of exposure time
    grid_scan_omega_range : float, optional
        Omega range of the grid scan in degrees, by default 0
    grid_scan_count_time : float, optional
        Detector count time. If this parameter is not defined, it is set to
        grid_scan_exposure_time - 0.0000001 by default
    screening_number_of_frames : int, optional
        The number of frames triggered during the screening step
    screening_start_omega: float, optional
        The screening start angle in degrees, by default 0
    screening_omega_range: float = 5, optional
        The screening omega range in degrees, by default
    screening_exposure_time: float = 1, optional
        The screening  exposure time in seconds, by default 1.
        NOTE: This is NOT the md3 definition of exposure time
    screening_number_of_passes: int = 1, optional
        The screening number of passes, by default 1
    screening_count_time : float, optional
        Detector count time. If this parameter is not defined, it is set to
        screening_exposure_time - 0.0000001 by default
    mount_pin_at_start_of_flow : bool, optional
        Mounts a pin at the start of the flow, by default False
    unmount_pin_when_flow_ends : bool, optional
        Unmounts a pin at the end of the flow, by default False
    hardware_trigger : bool, optional
        If set to true, we trigger the detector via hardware trigger, by default True.
        Warning! hardware_trigger=False is used only for development purposes.

    Returns
    -------
    None
    """
    if mount_pin_at_start_of_flow:
        await mount_pin(pin_id=pin_id, puck=puck)

    await optical_centering(
        sample_id=sample_id, beam_position=beam_position, grid_step=grid_step
    )

    async with asyncio.TaskGroup() as tg:
        _grid_scan_flat = tg.create_task(
            grid_scan_flat(
                sample_id=sample_id,
                exposure_time=grid_scan_exposure_time,
                omega_range=grid_scan_omega_range,
                count_time=grid_scan_count_time,
                hardware_trigger=hardware_trigger,
            )
        )
        crystal_finder_results_flat = tg.create_task(find_crystals(sample_id, "flat"))

        while not _grid_scan_flat.done():
            await asyncio.sleep(0.1)

        tg.create_task(
            grid_scan_edge(
                sample_id=sample_id,
                exposure_time=grid_scan_exposure_time,
                omega_range=grid_scan_omega_range,
                count_time=grid_scan_count_time,
                hardware_trigger=hardware_trigger,
            )
        )
        crystal_finder_results_edge = tg.create_task(find_crystals(sample_id, "edge"))

    crystals_flat = crystal_finder_results_flat.result()
    crystals_edge = crystal_finder_results_edge.result()

    if crystals_flat[0] is not None and crystals_edge[0] is not None:
        crystal_volumes, centers_of_mass = await find_crystals_in_3D(
            flat_coordinates=crystals_flat[0],
            edge_coordinates=crystals_edge[0],
            distance_flat=crystals_flat[1],
            distance_edge=crystals_edge[1],
            sample_id=sample_id,
            beam_position=beam_position,
        )

        for center_of_mass in centers_of_mass:
            if center_of_mass is not None:
                await screen_crystal(
                    sample_id=sample_id,
                    crystal_position=center_of_mass,
                    start_omega=screening_start_omega,
                    number_of_frames=screening_number_of_frames,
                    scan_range=screening_omega_range,
                    exposure_time=screening_exposure_time,
                    number_of_passes=screening_number_of_passes,
                    count_time=screening_count_time,
                    hardware_trigger=hardware_trigger,
                )

    if unmount_pin_when_flow_ends:
        await unmount_pin()


if __name__ == "__main__":
    asyncio.run(
        single_loop_data_collection(
            sample_id="my_sample",
            pin_id=3,
            puck=2,
            beam_position=(640, 512),
            grid_step=(81, 81),
            grid_scan_exposure_time=0.02,
            hardware_trigger=True,
            screening_omega_range=90,
        )
    )
