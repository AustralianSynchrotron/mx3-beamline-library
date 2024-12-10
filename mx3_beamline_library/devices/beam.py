from ophyd import EpicsSignal, EpicsSignalRO

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

transmission = EpicsSignal(
    "MX3FLT05:TransmissionSet",
    name="transmission",
)

ring_current = EpicsSignalRO(
    "SR11BCM01:CURRENT_MONITOR",
    name="ring_current",
)
