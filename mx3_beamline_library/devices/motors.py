""" Motor configuration and instantiation. """

from .classes.motors import MicroDiffractometer, MxcubeSimulatedPVs, Testrig

testrig = Testrig("TEST", name="testrig")

md3 = MicroDiffractometer()

mxcube_sim_PVs = MxcubeSimulatedPVs("MXCUBE", name="mxcube_sim_PVs")
