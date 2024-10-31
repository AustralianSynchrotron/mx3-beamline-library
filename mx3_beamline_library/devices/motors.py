""" Motor configuration and instantiation. """

from ophyd import EpicsSignalRO

from .classes.motors import ASBrickMotor, MicroDiffractometer
from .classes.robot import IsaraRobot

md3 = MicroDiffractometer()

isara_robot = IsaraRobot(name="robot")

detector_fast_stage = ASBrickMotor("MX3STG03MOT04", name="detector_fast_stage")

detector_slow_stage = ASBrickMotor("MX3STG03MOT01", name="detector_slow_stage")

actual_sample_detector_distance = EpicsSignalRO(
    "MX3ES01:SAMPLE_DETECTOR_DISTANCE", name="actual_sample_detector_distance"
)
