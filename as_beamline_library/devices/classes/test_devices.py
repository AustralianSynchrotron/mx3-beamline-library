from ophyd import MotorBundle, Component as Cpt
from ophyd.sim import SynAxis

class SimTable2(MotorBundle):
    """A Simulated Table."""

    x = Cpt(SynAxis, name="blah:X")
    y = Cpt(SynAxis, name="blah:Y")
    z = Cpt(SynAxis, name="blah:Z")

