""" Simulated motor configuration and instantiation. """
from .classes.motors import MySimTable, SimulatedPVs

# Simulated testrig motors
testrig = MySimTable(name="testrig")

mxcube_sim_PVs = SimulatedPVs(name="mxcube_sim_PVs")
