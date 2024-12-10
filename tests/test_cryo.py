from mx3_beamline_library.devices.cryo import cryo_temperature

def test_cryo_temperature():
    assert cryo_temperature.get() == 270.0