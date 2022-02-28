""" Example plans. See as-beamline-library-examples for some working examples. """

from functools import partial

# import bluesky.plan_stubs as bps
from bluesky.plans import scan

from ..devices.detectors import my_detector
from ..devices.motors import my_motor

step_shoot = partial(scan, [my_detector], my_motor)

# step_shoot_special_trigger = partial(scan, [my_detector], my_motor)