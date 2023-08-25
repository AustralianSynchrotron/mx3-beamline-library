from ...devices.classes.detectors import GrasshopperCamera, HDF5Filewriter
import uuid
from bluesky.plans import scan
from bluesky.plan_stubs import mv, read

def mx3_1d_scan(
    detectors: list[GrasshopperCamera], 
    motor,
    initial_position,
    final_position,
    number_of_steps,
    num=None, 
    per_step=None, 
    metadata: dict=None, 
    ):

    for detector in detectors:
        detector.kind = "hinted"

    hdf5_filename = str(uuid.uuid4()) + ".h5"

    if metadata is None:
        metadata = {"hdf5_filename": hdf5_filename}
    elif type(metadata) is dict:
        metadata.update({"hdf5_filename": hdf5_filename})
    elif type(metadata) is not dict:
        raise ValueError("Metadata must be a dictionary")
    
    for detector in detectors:
        if type(detector) == HDF5Filewriter:
            number_of_frames = number_of_steps
            yield from mv(
                detector.filename, 
                hdf5_filename, 
                detector.frames_per_datafile,
                number_of_frames
            )
            write_path_template = detector.write_path_template.get()
            metadata.update({"write_path_template": write_path_template})
    yield from scan(detectors, motor, initial_position, final_position, number_of_steps,num=num, per_step=per_step, md=metadata)
