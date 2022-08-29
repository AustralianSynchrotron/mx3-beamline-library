""" Motor configuration and instantiation. """

from .classes.motors import MxcubeSimulatedPVs, Testrig

testrig = Testrig("TEST", name="testrig")

mxcube_sim_PVs = MxcubeSimulatedPVs("MXCUBE", name="mxcube_sim_PVs")
