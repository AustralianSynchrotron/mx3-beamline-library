import tiled
from os import path
from tiled.queries import Contains


def get_hdf5_file_path_from_bluesky_scan_id(tiled_client: tiled.client.constructors.from_uri, bluesky_scan_id: str):
    start_document = tiled_client[bluesky_scan_id].metadata["start"]
    hdf5_file_path = path.join(start_document["write_path_template"], start_document["hdf5_filename"])
    return hdf5_file_path

def get_bluesky_scan_id_from_hdf5_filename(
        tiled_client: tiled.client.constructors.from_uri, 
        hdf5_filename: str): #returns bluesky run
    # hdf5_filename can be either the full path or just the filename
    
    value = path.basename(path.normpath(hdf5_filename))
    bluesky_run = tiled_client.search(Contains(
        key="hdf5_filename", value=value))[0]

    return bluesky_run


