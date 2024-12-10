from mx3_beamline_library.devices.beam import energy_master, transmission, attenuation, ring_current, energy_dmm


def test_energy_master():
    assert energy_master.get() == 13.0
    energy_master.set(13.2)
    assert energy_master.get() == 13.2

def test_transmission():
    assert transmission.get() == 0.1
    transmission.set(0.2)
    assert transmission.get() == 0.2

def test_attenuation():
    assert attenuation.get() == 0.9
    attenuation.set(0.8)
    assert attenuation.get() == 0.8

def test_energy_dmm():
    assert energy_dmm.get() == 13.0
    energy_dmm.set(13.2)
    assert energy_dmm.get() == 13.2

def test_ring_current():
    assert ring_current.get() == 200.0

