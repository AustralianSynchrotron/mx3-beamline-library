"""
This example shows how to mount and unmount samples using the bluesky run engine
"""
from os import environ
environ["ROBOT_HOST"] = "12.345.678.9" # noqa

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from mx3_beamline_library.devices.motors import isara_robot
from mx3_beamline_library.plans.robot import mount_pin, unmount_pin



print(isara_robot.state.get())

RE = RunEngine()
bec = BestEffortCallback()
RE.subscribe(bec)

mount_signal = isara_robot.mount
unmount_signal = isara_robot.unmount

# Mount pin
RE(mount_pin(mount_signal, id=1, puck=1))

# Unmount pin
RE(unmount_pin(unmount_signal))
