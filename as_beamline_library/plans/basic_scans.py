""" Example plans. """

from functools import partial

import bluesky.plan_stubs as bps
from bluesky.plans import scan

from as_beamline_library.devices.detectors import my_detector
from as_beamline_library.devices.motors import my_motor

step_shoot = partial(scan, [my_detector], my_motor)
