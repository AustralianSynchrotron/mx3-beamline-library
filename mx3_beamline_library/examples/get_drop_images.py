"""
This examples shows how to take snapshots of drops at positions specified
in the drop_location list
"""
from os import environ

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback

environ["BL_ACTIVE"] = "False"
environ["MD3_ADDRESS"] = "12.345.678.90"
environ["MD3_PORT"] = "9001"
environ["MD_REDIS_HOST"] = "12.345.678.90"
environ["MD_REDIS_PORT"] = "6379"

from mx3_beamline_library.plans.tray_scans import save_drop_snapshots  # noqa

# Instantiate run engine an start plan
RE = RunEngine({})
bec = BestEffortCallback()
RE.subscribe(bec)

RE(
    save_drop_snapshots(
        tray_id="my_tray",
        drop_locations=["B1-1", "C1-1"],
        alignment_y_offset=0.25,
        alignment_z_offset=-1.0,
    )
)
