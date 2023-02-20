""" Simulated motor configuration and instantiation. """
from .classes.motors import MySimTable, SimulatedPVs, SimMicroDiffractometer

# Simulated testrig motors
testrig = MySimTable(name="testrig")

md3 = SimMicroDiffractometer(name="sim")

mxcube_sim_PVs = SimulatedPVs(name="mxcube_sim_PVs")
