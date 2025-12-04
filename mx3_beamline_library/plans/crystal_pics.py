import pickle
from io import BytesIO
from typing import Literal
from uuid import UUID

import h5py
import numpy as np
import numpy.typing as npt
from PIL import Image

from ..config import redis_connection
from ..schemas.xray_centering import RasterGridCoordinates
from .image_analysis import get_image_from_md3_camera


def get_md3_camera_jpeg_image() -> bytes:
    """
    Gets a numpy array from the md3 camera and stores it as a JPEG image
    using the io and PIL libraries

    Returns
    -------
    bytes
        The md3 camera image in bytes format
    """
    array = get_image_from_md3_camera("uint8")
    pil_image = Image.fromarray(array)

    with BytesIO() as f:
        pil_image.save(f, format="JPEG")
        jpeg_image = f.getvalue()

    return jpeg_image


def save_crystal_pic_to_redis(
    acquisition_uuid: UUID,
    collection_stage: Literal["start", "end"],
    expiry: float | None = 3600,
) -> None:
    """
    Saves a crystal picture to redis. The key is generated
    based on the acquisition_uuid and collection_stage.

    Parameters
    ----------
    acquisition_uuid : UUID
        The acquisition uuid
    collection_stage : Literal["start", "end"]
        The collection stage (start or end)
    expiry : float | None, optional
        The expiry, by default 3600

    Raises
    ------
    ValueError
        Raises an error if the type of data collection is not supported
    """
    key = f"crystal_pic_{collection_stage}:{acquisition_uuid}"

    redis_connection.set(key, get_md3_camera_jpeg_image(), ex=expiry)


def get_screen_or_dataset_crystal_pic(
    acquisition_uuid: UUID,
    collection_stage: Literal["start", "end"],
) -> npt.NDArray:
    """
    Gets a crystal picture from redis and returns a numpy array.

    Parameters
    ----------
    acquisition_uuid : UUID
        The acquisition uuid
    collection_stage : Literal["start", "end"]
        The collection stage (start or end)

    Returns
    -------
    npt.NDArray
        The crystal picture as a numpy array
    """
    key = f"crystal_pic_{collection_stage}:{acquisition_uuid}"

    result = redis_connection.get(key)
    image_array = Image.open(BytesIO(result))
    return np.array(image_array)


def add_crystal_pic_to_hdf5(
    hdf5_file: str,
    acquisition_uuid: UUID,
) -> None:

    start = get_screen_or_dataset_crystal_pic(
        acquisition_uuid=acquisition_uuid,
        collection_stage="start",
    )
    end = get_screen_or_dataset_crystal_pic(
        acquisition_uuid=acquisition_uuid,
        collection_stage="end",
    )

    with h5py.File(hdf5_file, mode="r+") as hf:
        hf.create_dataset("entry/xtal_pic/start", data=start, compression="lzf")
        hf.create_dataset("entry/xtal_pic/end", data=end, compression="lzf")


def get_grid_scan_crystal_pic(
    acquisition_uuid: UUID | None = None,
    grid_scan_id: Literal["flat", "edge"] | None = None,
    sample_id: str | None = None,
) -> npt.NDArray:
    """
    Gets grid-scan crystal picture from redis and return it as a numpy array.

    UDC and mxcube are supported
    - UDC requires sample_id and grid_scan_id
    - MXCuBE requires acquisition_uuid.

    Parameters
    ----------
    acquisition_uuid : UUID | None
        The acquisition UUID (required for MXCuBE snapshots)
    grid_scan_id : Literal["flat", "edge"] | None
        The UDC grid scan identifier (requires sample_id)
    sample_id : str | None
        The sample identifier used for UDC results

    Returns
    -------
    npt.NDArray
        The crystal picture as a numpy array
    """
    # UDC (flat or edge) require a sample_id
    if grid_scan_id in ["flat", "edge"]:
        if not sample_id:
            raise ValueError(
                "sample_id is required when grid_scan_id is 'flat' or 'edge'"
            )

        r = redis_connection.get(f"optical_centering_results:{sample_id}")
        if r is None:
            raise ValueError(
                f"No optical centering results found for sample {sample_id}"
            )

        results = pickle.loads(r)
        key = f"{grid_scan_id}_grid_motor_coordinates"
        if key not in results:
            raise ValueError(
                f"Key '{key}' not found in optical centering results for sample {sample_id}"
            )

        coords = RasterGridCoordinates.model_validate(results[key])
        img = Image.open(BytesIO(coords.md3_camera_snapshot))
        return np.array(img)

    # MXCuBE only requires acquisition_uuid
    if acquisition_uuid is None:
        raise ValueError(
            "acquisition_uuid is required when grid_scan_id is not 'flat' or 'edge'"
        )

    data = redis_connection.get(f"mxcube:grid_scan_snapshot:{acquisition_uuid}")
    if data is None:
        raise ValueError(
            f"No MXCuBE grid scan snapshot for acquisition {acquisition_uuid}"
        )
    img = Image.open(BytesIO(data))
    return np.array(img)


def save_mxcube_grid_scan_crystal_pic(
    acquisition_uuid: UUID,
) -> None:
    """
    Saves a crystal picture of grid scans to redis

    Parameters
    ----------
    acquisition_uuid : UUID
        The acquisition uuid
    """
    redis_connection.set(
        f"mxcube:grid_scan_snapshot:{acquisition_uuid}",
        get_md3_camera_jpeg_image(),
    )
