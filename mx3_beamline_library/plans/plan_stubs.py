# from ..devices.classes.motors import SERVER, MD3Motor
from mx3_beamline_library.devices.classes.motors import SERVER, MD3Motor
from bluesky.utils import Msg, merge_cycler
import uuid
from cycler import cycler
import operator
from functools import reduce
from time import sleep
from os import environ
from bluesky.plan_stubs import mv
try:
    # cytools is a drop-in replacement for toolz, implemented in Cython
    from cytools import partition
except ImportError:
    from toolz import partition
    

def md3_move(*args, group=None, **kwargs):
    """
    Move one or more md3 motors to a setpoint. Wait for all to complete.
    If more than one device is specified, the movements are done in parallel.
    
    Parameters
    ----------
    args :
        device1, value1, device2, value2, ...
    group : string, optional
        Used to mark these as a unit to be waited on.
    kwargs :
        passed to obj.set()
    
    Yields
    ------
    msg : Msg

    """
    group = group or str(uuid.uuid4())

    cyl = reduce(operator.add, [cycler(obj, [val]) for obj, val in partition(2, args)])
    (step,) = merge_cycler(cyl)
    
    cmd = str()
    for obj, val in step.items():
        cmd += f"{obj.name}={val},"
    
    if environ["BL_ACTIVE"].lower() == "true":
        SERVER.startSimultaneousMoveMotors(
            cmd
            )
        status = "Running"
        while status == "Running":
            status = SERVER.getState()
            sleep(0.1)
        yield Msg('wait', None, group=group)
    else:
        yield from mv(*args)



if __name__ == "__main__":
    from bluesky import RunEngine
    from bluesky.callbacks.best_effort import BestEffortCallback
    from mx3_beamline_library.devices.motors import md3

    # Instantiate run engine an start plan
    RE = RunEngine({})
    bec = BestEffortCallback()
    RE.subscribe(bec)

    RE(
        md3_move(md3.sample_x, 0.0, md3.sample_y,0.0, md3.alignment_x, 0.0, md3.alignment_y, 0.0, md3.alignment_z, 0.0, md3.plate_translation, 10)
    )
