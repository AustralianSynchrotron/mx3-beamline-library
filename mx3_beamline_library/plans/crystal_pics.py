import pickle
from io import BytesIO
from typing import Literal

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


def save_screen_or_dataset_crystal_pic_to_redis(
    sample_id: int | str,
    crystal_counter: int,
    data_collection_counter: int,
    type: Literal["screening", "dataset"],
    collection_stage: Literal["start", "end"],
    expiry: float | None = 3600,
) -> None:
    """
    Saves a crystal picture to redis. The key is generated based on the
    sample_id, crystal_counter, data_collection_counter and type of
    collection (screening or dataset). The key is also generated based on
    the collection stage (start or end)

    Parameters
    ----------
    sample_id : int | str
        The sample id
    crystal_counter : int
        The crystal counter
    data_collection_counter : int
        The data collection counter
    type : Literal["screening", "dataset"]
        The type of data collection
    collection_stage : Literal["start", "end"]
        The collection stage (start or end)
    expiry : float | None, optional
        The expiry, by default 3600

    Raises
    ------
    ValueError
        Raises an error if the type of data collection is not supported
    """
    if type == "screening":
        key = f"screening_pic:{collection_stage}:sample_{sample_id}:crystal_{crystal_counter}:data_collection_{data_collection_counter}"  # noqa
    elif type == "dataset":
        key = f"dataset_pic:{collection_stage}:sample_{sample_id}:crystal_{crystal_counter}:data_collection_{data_collection_counter}"  # noqa
    else:
        raise ValueError(f"Data collection type {type} not supported")

    redis_connection.set(key, get_md3_camera_jpeg_image(), ex=expiry)


def get_screen_or_dataset_crystal_pic(
    sample_id: int | str,
    crystal_counter: int,
    data_collection_counter: int,
    type: Literal["screening", "dataset"],
    collection_stage: Literal["start", "end"],
) -> npt.NDArray:
    """
    Gets a crystal picture from redis and returns a numpy array.

    Parameters
    ----------
    sample_id : int | str
        The sample id
    crystal_counter : int
        The crystal counter
    data_collection_counter : int
        The data collection counter
    type : Literal["screening", "dataset"]
        The type of data collection
    collection_stage : Literal["start", "end"]
        The collection stage (start or end)



    Returns
    -------
    npt.NDArray
        The crystal picture as a numpy array

    Raises
    ------
    ValueError
        Raises an error if the type of data collection is not supported
    """
    if type == "screening":
        key = f"screening_pic:{collection_stage}:sample_{sample_id}:crystal_{crystal_counter}:data_collection_{data_collection_counter}"  # noqa
    elif type == "dataset":
        key = f"dataset_pic:{collection_stage}:sample_{sample_id}:crystal_{crystal_counter}:data_collection_{data_collection_counter}"  # noqa
    else:
        raise ValueError(f"Data collection type {type} not supported")

    result = redis_connection.get(key)
    image_array = Image.open(BytesIO(result))
    return np.array(image_array)


def add_crystal_pic_to_hdf5(
    hdf5_file: str,
    sample_id: int | str,
    crystal_counter: int,
    data_collection_counter: int,
    type: Literal["screening", "dataset"],
) -> None:
    if type == "screening":
        start = get_screen_or_dataset_crystal_pic(
            sample_id=sample_id,
            crystal_counter=crystal_counter,
            data_collection_counter=data_collection_counter,
            type=type,
            collection_stage="start",
        )
        end = get_screen_or_dataset_crystal_pic(
            sample_id=sample_id,
            crystal_counter=crystal_counter,
            data_collection_counter=data_collection_counter,
            type=type,
            collection_stage="end",
        )
    elif type == "dataset":
        start = get_screen_or_dataset_crystal_pic(
            sample_id=sample_id,
            crystal_counter=crystal_counter,
            data_collection_counter=data_collection_counter,
            type=type,
            collection_stage="start",
        )
        end = get_screen_or_dataset_crystal_pic(
            sample_id=sample_id,
            crystal_counter=crystal_counter,
            data_collection_counter=data_collection_counter,
            type=type,
            collection_stage="end",
        )
    else:
        raise ValueError(f"Data collection type {type} not supported")

    with h5py.File(hdf5_file, mode="r+") as hf:
        hf.create_dataset("entry/xtal_pic/start", data=start, compression="lzf")
        hf.create_dataset("entry/xtal_pic/end", data=end, compression="lzf")


def get_grid_scan_crystal_pic(
    sample_id: int | str, grid_scan_id: int | str
) -> npt.NDArray:
    """
    Gets crystal pictures of grid scans from redis and returns
    a numpy array

    Parameters
    ----------
    sample_id : int | str
        The sample id
    grid_scan_id : int | str
        The grid scan id

    Returns
    -------
    npt.NDArray
        The crystal picture
    """
    if grid_scan_id == "flat":
        # UDC
        r = redis_connection.get(f"optical_centering_results:{sample_id}")
        if r is None:
            raise ValueError(
                f"No results found for sample {sample_id}, grid scan {grid_scan_id}"
            )
        results = pickle.loads(r)

        flat_grid_coordinates = RasterGridCoordinates.model_validate(
            results["flat_grid_motor_coordinates"]
        )
        flat = Image.open(BytesIO(flat_grid_coordinates.md3_camera_snapshot))
        return np.array(flat)

    elif grid_scan_id == "edge":
        # UDC
        r = redis_connection.get(f"optical_centering_results:{sample_id}")
        if r is None:
            raise ValueError(
                f"No results found for sample {sample_id}, grid_scan_id {grid_scan_id}"
            )
        results = pickle.loads()
        edge_grid_coordinates = RasterGridCoordinates.model_validate(
            results["edge_grid_motor_coordinates"]
        )
        edge = Image.open(BytesIO(edge_grid_coordinates.md3_camera_snapshot))
        return np.array(edge)

    else:
        # MXCuBE
        results = redis_connection.get(
            f"mxcube_grid_scan_snapshot_{grid_scan_id}:{sample_id}"
        )
        if results is None:
            raise ValueError(
                f"No results found for sample {sample_id}, grid scan {grid_scan_id}"
            )
        mxcube = Image.open(BytesIO(results))
        return np.array(mxcube)
