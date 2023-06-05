""" Simulated motor configuration and instantiation. """
from .classes.motors import IsaraRobot, MySimTable, SimMicroDiffractometer, SimulatedPVs

# Simulated testrig motors
testrig = MySimTable(name="testrig")

md3 = SimMicroDiffractometer(name="md3")

mxcube_sim_PVs = SimulatedPVs(name="mxcube_sim_PVs")

isara_robot = IsaraRobot(name="robot")
