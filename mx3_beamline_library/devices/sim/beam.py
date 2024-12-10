from ophyd import Signal

energy_master = Signal(name="energy_master", value=13.0)

energy_dmm = Signal(name="energy_dmm", value=13.0)

attenuation = Signal(name="attenuation", value=0.9)

transmission = Signal(name="transmission", value=0.1)

ring_current = Signal(name="ring_current", value=200.0)
