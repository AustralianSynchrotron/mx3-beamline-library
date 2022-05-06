from ophyd import EpicsSignal
from ..constants import PV_PREFIX


exp_shutter = EpicsSignal(f"{PV_PREFIX}:EXP_SHUTTER", name="Experimental Shutter")
