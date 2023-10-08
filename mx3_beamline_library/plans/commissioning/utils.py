from os import path

import tiled
from tiled.queries import Contains


def get_hdf5_file_path_from_bluesky_scan_id(
    tiled_client: tiled.client.constructors.from_uri, bluesky_scan_id: str
) -> str:
    """
    Gets the hdf5 file path from the bluesky scan id

    Parameters
    ----------
    tiled_client : tiled.client.constructors.from_uri
        A tiled client
    bluesky_scan_id : str
        The bluesky unique scan id

    Returns
    -------
    str
       The path of the hdf5 file generated during the run
    """
    start_document = tiled_client[bluesky_scan_id].metadata["start"]
    hdf5_file_path = path.join(
        start_document["write_path_template"], start_document["hdf5_filename"]
    )
    return hdf5_file_path


def get_bluesky_scan_id_from_hdf5_filename(
    tiled_client: tiled.client.constructors.from_uri, hdf5_filename: str
):
    """
    Gets the bluesky documents from the name of a hdf5 file.
    The name of the hdf5 file can include either the pull path, or the
    name of the file, e.g., /path/to/file/file.h5 or file.h5

    Parameters
    ----------
    tiled_client : tiled.client.constructors.from_uri
        A tiled client
    hdf5_filename : str
        The name of the hdf5 file. Can include the full path,
        or just the name of the file.

    Returns
    -------
        The bluesky documents associated with a hdf5 file
    """

    value = path.basename(path.normpath(hdf5_filename))
    bluesky_run = tiled_client.search(Contains(key="hdf5_filename", value=value))[0]

    return bluesky_run
