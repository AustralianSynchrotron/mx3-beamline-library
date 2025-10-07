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

# PVs used for beam steering
ROOT = "MX3DAQIOC04:"
# Change control. 1 is EPICS 0 is FPGA
control = EpicsSignal(ROOT + "PreDAC0:OutMux", name="control")


# syntax is "ON" or "OFF"
steering_enable = EpicsSignal(ROOT + "PID:Enable", name="steering_enable")

# Output channels. 1 is X and 2 is Y. Value is volts as float e.g. "1.2"
x_Volt_SP = EpicsSignal(ROOT + "PreDAC0:OutCh2", name="x_Volt_SP")
y_Volt_SP = EpicsSignal(ROOT + "PreDAC0:OutCh1", name="y_Volt_SP")

# voltage readbacks (volts)
x_Volt_RBV = EpicsSignal(ROOT + "PreDAC0:Ch2_RBV", name="x_Volt_RBV")
y_Volt_RBV = EpicsSignal(ROOT + "PreDAC0:Ch1_RBV", name="y_Volt_RBV")

# Flux readback (Amps)

# Setpoints
x_SP = EpicsSignal(ROOT + "PID:SetpointX", name="x_SP")
y_SP = EpicsSignal(ROOT + "PID:SetpointY", name="y_SP")

# Position Readback
x_RBV = EpicsSignal(ROOT + "BPM0:PosX_RBV", name="x_RBV")
y_RBV = EpicsSignal(ROOT + "BPM0:PosY_RBV", name="y_RBV")

# flux_readback
flux_beam_steering = EpicsSignal(ROOT + "BPM0:Int_RBV", name="flux")

# BeamOffThreshold
beamOffThreshold_SP = EpicsSignal(ROOT + "BPM0:BeamOffTh", name="beamOffThreshold_SP")
beamOffThreshold_RBV = EpicsSignal(
    ROOT + "BPM0:BeamOffTh_RBV", name="beamOffThreshold_RBV"
)

kill_goni_lateral = EpicsSignal(
    "MX3STG02MOT01:STOP_KILL.PROC", name="kill_goni_lateral"
)
kill_goni_vertical = EpicsSignal(
    "MX3STG02MOT02:STOP_KILL.PROC", name="kill_goni_vertical"
)


# Position Readback
x_RBV = EpicsSignal(ROOT + "BPM0:PosX_RBV", name="x_RBV")
y_RBV = EpicsSignal(ROOT + "BPM0:PosY_RBV", name="y_RBV")
