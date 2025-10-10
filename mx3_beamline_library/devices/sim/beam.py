from ophyd import Signal

from .classes.beam import BPM

energy_master = Signal(name="energy_master", value=13.0)

energy_dmm = Signal(name="energy_dmm", value=13.0)

attenuation = Signal(name="attenuation", value=0.9)

transmission = Signal(name="transmission", value=0.1)

filter_wheel_is_moving = Signal(name="filter_wheel_is_moving", value=0)

ring_current = Signal(name="ring_current", value=200.0)

dmm_stripe = Signal(name="dmm_stripe", value=1)

flux = Signal(name="flux", value=2.30e11)

# PVs used for beam steering
bmp_1 = BPM(name="bmp_1")

kill_goni_lateral = Signal(name="kill_goni_lateral", value=0)
kill_goni_vertical = Signal(name="kill_goni_vertical", value=0)
