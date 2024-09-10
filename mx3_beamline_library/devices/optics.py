from ophyd import EpicsSignal

energy_master = EpicsSignal("MX3:MASTER_ENERGY_SP", name="energy_master")

energy_dmm = EpicsSignal(
    read_pv="MX3MONO01:ENERGY_MONITOR",
    write_pv="MX3MONO01:ENERGY_SP",
    name="energy_dmm",
)

attenuation = EpicsSignal(
    read_pv="MX3FLT05:AttenuationRBV",
    write_pv="MX3FLT05:AttenuationSet",
    name="attenuation",
)
