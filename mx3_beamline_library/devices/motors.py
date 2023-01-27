""" Motor configuration and instantiation. """

from .classes.motors import MicroDiffractometer, MxcubeSimulatedPVs

md3 = MicroDiffractometer()

mxcube_sim_PVs = MxcubeSimulatedPVs("MXCUBE", name="mxcube_sim_PVs")
