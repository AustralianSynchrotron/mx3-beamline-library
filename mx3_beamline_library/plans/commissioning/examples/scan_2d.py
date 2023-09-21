from bluesky import RunEngine
from bluesky.plans import grid_scan, scan
from ophyd.sim import motor1, det1, motor2, motor3
import numpy as np
import os
os.environ["BL_ACTIVE"]="True"
os.environ["EPICS_CA_ADDR_LIST"] = "10.244.101.20 10.244.101.1"

from ophyd.areadetector.base import DDC_EpicsSignalRO

from mx3_beamline_library.devices.classes.detectors import GrasshopperCamera, HDF5Filewriter
from mx3_beamline_library.plans.commissioning.commissioning import Scan2D



my_cam = GrasshopperCamera("13SIM1", name="blackfly_camera")

motor1.delay = 0.001
motor2.delay = 0.001


RE = RunEngine({})
print(my_cam.stats.total.get())
#RE.subscribe(tiled_client.post_document)

scan_2d = Scan2D([my_cam.stats.total],
        motor1, -1,100, 5, motor2, -1,100,5)
RE(scan_2d.run())


#my_cam.file_plugin.write_file.trigger()


