from ophyd import Device, Signal


class BPM(Device):
    """
    Simulated BPM, only used for testing purposes
    """

    steering_enable = Signal(name="steering_enable")
    control = Signal(name="control")
    x_volt = Signal(name="x_volt")
    y_volt = Signal(name="y_volt")
    x = Signal(name="x")
    y = Signal(name="y")
    flux = Signal(name="flux")
    beam_off_threshold = Signal(name="beam_off_threshold")
