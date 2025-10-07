from ophyd import Signal

energy_master = Signal(name="energy_master", value=13.0)

energy_dmm = Signal(name="energy_dmm", value=13.0)

attenuation = Signal(name="attenuation", value=0.9)

transmission = Signal(name="transmission", value=0.1)

filter_wheel_is_moving = Signal(name="filter_wheel_is_moving", value=0)

ring_current = Signal(name="ring_current", value=200.0)

dmm_stripe = Signal(name="dmm_stripe", value=1)

flux = Signal(name="flux", value=2.30e11)

# PVs used for beam steering
control = Signal(name="control", value=0)


# syntax is "ON" or "OFF"
steering_enable = Signal(name="steering_enable", value=0)

# Output channels. 1 is X and 2 is Y. Value is volts as float e.g. "1.2"
x_Volt_SP = Signal(name="x_Volt_SP", value=0)
y_Volt_SP = Signal(name="y_Volt_SP", value=0)

# voltage readbacks (volts)
x_Volt_RBV = Signal(name="x_Volt_RBV", value=0)
y_Volt_RBV = Signal(name="y_Volt_RBV", value=0)

# Flux readback (Amps)

# Setpoints
x_SP = Signal(name="x_SP", value=0)
y_SP = Signal(name="y_SP", value=0)

# Position Readback
x_RBV = Signal(name="x_RBV", value=0)
y_RBV = Signal(name="y_RBV", value=0)

# flux_readback
flux_beam_steering = Signal(name="flux_beam_steering", value=0)

# BeamOffThreshold
beamOffThreshold_SP = Signal(name="beamOffThreshold_SP", value=0)
beamOffThreshold_RBV = Signal(name="beamOffThreshold_RBV", value=0)

kill_goni_lateral = Signal(name="kill_goni_lateral", value=0)
kill_goni_vertical = Signal(name="kill_goni_vertical", value=0)


# Position Readback
x_RBV = Signal(name="x_RBV", value=0)
y_RBV = Signal(name="y_RBV", value=0)
