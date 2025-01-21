import pytest
from bluesky.plan_stubs import mv
from bluesky.run_engine import RunEngine

from mx3_beamline_library.devices.beam import (
    attenuation,
    energy_dmm,
    energy_master,
    ring_current,
    transmission,
)


@pytest.mark.order("first")
def test_energy_master(run_engine: RunEngine):
    assert energy_master.get() == 13.0
    run_engine(mv(energy_master, 13.2))
    assert energy_master.get() == 13.2


@pytest.mark.order(after="test_energy_master")
def test_transmission(run_engine: RunEngine):
    assert transmission.get() == 0.1
    run_engine(mv(transmission, 0.2))
    assert transmission.get() == 0.2


@pytest.mark.order(after="test_transmission")
def test_attenuation(run_engine: RunEngine):
    assert attenuation.get() == 0.9
    run_engine(mv(attenuation, 0.8))
    assert attenuation.get() == 0.8


@pytest.mark.order(after="test_attenuation")
def test_energy_dmm(run_engine: RunEngine):
    assert energy_dmm.get() == 13.0
    run_engine(mv(energy_dmm, 13.2))
    assert energy_dmm.get() == 13.2


@pytest.mark.order(after="test_energy_dmm")
def test_ring_current():
    assert ring_current.get() == 200.0
