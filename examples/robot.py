"""
This example shows how to mount and unmount samples using the bluesky run engine
    Requirements:
    - Access to the ISARA robot server
"""
from os import environ

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback

environ["ROBOT_HOST"] = "12.345.678.9"  # Add the robot host here
environ["BL_ACTIVE"] = "True"


from mx3_beamline_library.plans.robot import mount_pin, unmount_pin  # noqa

RE = RunEngine()
bec = BestEffortCallback()
RE.subscribe(bec)

# Mount pin
RE(mount_pin(id=4, puck=2))

# Unmount pin
RE(unmount_pin())
