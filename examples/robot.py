"""
This example shows how to mount and unmount samples using the bluesky run engine
"""
from os import environ

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback

environ["ROBOT_HOST"] = "12.345.678.9"  # Add the robot host here
environ["BL_ACTIVE"] = "True"


from mx3_beamline_library.devices.motors import isara_robot  # noqa
from mx3_beamline_library.plans.robot import mount_pin, unmount_pin  # noqa

RE = RunEngine()
bec = BestEffortCallback()
RE.subscribe(bec)

mount_signal = isara_robot.mount
unmount_signal = isara_robot.unmount

# Mount pin
RE(mount_pin(mount_signal, id=1, puck=1))

# Unmount pin
RE(unmount_pin(unmount_signal))
