from ophyd import EpicsSignal, EpicsSignalRO

from .classes.beam import Transmission

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

transmission = Transmission(
    read_pv="MX3FLT05:TransmissionRBV",
    write_pv="MX3FLT05:TransmissionSet",
    is_moving_pv="MX3FLT05MOT01.MOVN",
    name="transmission",
)

filter_wheel_is_moving = EpicsSignalRO(
    "MX3FLT05MOT01.MOVN", name="filter_wheel_is_moving"
)

ring_current = EpicsSignalRO(
    "SR11BCM01:CURRENT_MONITOR",
    name="ring_current",
)

dmm_stripe = EpicsSignalRO("MX3MONO01:STRIPE_POSITION_MONITOR", name="dmm_stripe")

flux = EpicsSignalRO("MX3FLUXIOC:FLUX", name="flux")
