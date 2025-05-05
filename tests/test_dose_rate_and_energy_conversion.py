import pytest

from mx3_beamline_library.science.energy_conversion import (
    Angstrom_to_keV,
    keV_to_Angstrom,
)
from mx3_beamline_library.science.holton2009 import dose_rate


def test_energy_to_wavelength():
    # Test conversion from keV to Ångström
    energy_keV = 13.00
    expected_wavelength = 0.95372460
    assert expected_wavelength == pytest.approx(keV_to_Angstrom(energy_keV), rel=1e-6)


def test_wavelength_to_energy():
    # Test conversion from Ångström to keV
    wavelength_Ångström = 0.95372460
    expected_energy = 13.0
    assert expected_energy == pytest.approx(
        Angstrom_to_keV(wavelength_Ångström), rel=1e-6
    )


def test_dose_rate_success1():
    # Test dose rate calculation with wavelength
    flux = 1e12  # photons/s
    beam_width = 100  # micrometers
    beam_height = 100  # micrometers
    wavelength = 1
    expected_dose_rate = 0.05
    assert expected_dose_rate == pytest.approx(
        dose_rate(
            flux,
            beam_width,
            beam_height,
            wavelength=wavelength,
            energy=None,
        ),
        rel=1e-6,
    )


def test_dose_rate_success2():
    # Test dose rate calculation with energy
    flux = 1e12  # photons/s
    beam_width = 100  # micrometers
    beam_height = 100  # micrometers
    energy = 12.398419
    expected_dose_rate = 0.05
    assert expected_dose_rate == pytest.approx(
        dose_rate(
            flux,
            beam_width,
            beam_height,
            wavelength=None,
            energy=energy,
        ),
        rel=1e-6,
    )


def test_dose_rate_failure1():
    # Test dose rate calculation with both energy and wavelength
    flux = 1e12  # photons/s
    beam_width = 100  # micrometers
    beam_height = 100  # micrometers
    energy = 12.398419
    wavelength = 1
    with pytest.raises(ValueError):
        dose_rate(
            flux,
            beam_width,
            beam_height,
            wavelength=wavelength,
            energy=energy,
        )


def test_dose_rate_failure2():
    # Test dose rate calculation with neither energy nor wavelength
    flux = 1e12  # photons/s
    beam_width = 100  # micrometers
    beam_height = 100  # micrometers
    with pytest.raises(ValueError):
        dose_rate(
            flux,
            beam_width,
            beam_height,
            wavelength=None,
            energy=None,
        )
