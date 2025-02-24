"""Simulated motor configuration and instantiation."""

from ophyd import Signal

from .classes.motors import (
    IsaraRobot,
    MX3SimMotor,
    SimMicroDiffractometer,
    SimulatedPVs,
)

# Simulated testrig motors

md3 = SimMicroDiffractometer(name="md3")

mxcube_sim_PVs = SimulatedPVs(name="mxcube_sim_PVs")

isara_robot = IsaraRobot(name="robot")

detector_fast_stage = MX3SimMotor(name="detector_fast_stage")

detector_slow_stage = MX3SimMotor(name="detector_slow_stage")

actual_sample_detector_distance = Signal(
    name="actual_sample_detector_distance", value=100
)
