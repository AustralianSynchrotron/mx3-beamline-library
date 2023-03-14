""" Motor configuration and instantiation. """

from .classes.motors import MicroDiffractometer, MxcubeSimulatedPVs
from .classes.robot import IsaraRobot

md3 = MicroDiffractometer()

isara_robot = IsaraRobot(name="robot")

mxcube_sim_PVs = MxcubeSimulatedPVs("MXCUBE", name="mxcube_sim_PVs")
