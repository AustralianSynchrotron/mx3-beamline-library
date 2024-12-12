from mx3_beamline_library.devices.shutters import white_beam_shutter, mono_beam_shutter
from mx3_beamline_library.devices.classes.shutters import OpenCloseStatus

def test_white_beam_shutter():
    assert white_beam_shutter.open_close_status.get() == 2
    assert white_beam_shutter.open_close_status.get() == OpenCloseStatus.CLOSED
    assert white_beam_shutter.open_enabled.get() == 1
    white_beam_shutter.open_close_cmd.set(1) # close valve # TODO: test with real HW

def test_mono_beam_shutter():
    assert mono_beam_shutter.open_close_status.get() == 2
    assert white_beam_shutter.open_close_status.get() == OpenCloseStatus.CLOSED
    assert mono_beam_shutter.open_enabled.get() == 1
    mono_beam_shutter.open_close_cmd.set(1) # close valve # TODO: test with real HW