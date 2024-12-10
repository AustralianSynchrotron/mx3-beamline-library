from .classes.shutters import SinglePSSShutters

white_beam_shutter = SinglePSSShutters(
    "MX3FE01SHT01", name="white_beam_shutter"
)  # aka front end shutter

mono_beam_shutter = SinglePSSShutters("MX3BLSH01SHT01", name="mono_beam_shutter")
