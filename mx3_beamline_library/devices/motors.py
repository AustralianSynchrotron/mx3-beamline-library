""" Motor configuration and instantiation. """

from .classes.motors import MxcubeSimulatedPVs, Testrig

testrig = Testrig("MX3-testrig", name="testrig")

mxcube_sim_PVs = MxcubeSimulatedPVs("MXCUBE", name="mxcube_sim_PVs")
