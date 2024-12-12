from mx3_beamline_library.devices.classes.shutters import OpenCloseCmd, OpenCloseStatus
from mx3_beamline_library.devices.shutters import mono_beam_shutter, white_beam_shutter


def test_white_beam_shutter():
    assert white_beam_shutter.open_close_status.get() == 2
    assert white_beam_shutter.open_close_status.get() == OpenCloseStatus.CLOSED
    assert white_beam_shutter.open_enabled.get() == 1
    white_beam_shutter.open_close_cmd.set(
        OpenCloseCmd.CLOSE_VALVE
    )  # TODO: test with real HW


def test_mono_beam_shutter():
    assert mono_beam_shutter.open_close_status.get() == 2
    assert white_beam_shutter.open_close_status.get() == OpenCloseStatus.CLOSED
    assert mono_beam_shutter.open_enabled.get() == 1
    mono_beam_shutter.open_close_cmd.set(1)  # TODO: test with real HW
