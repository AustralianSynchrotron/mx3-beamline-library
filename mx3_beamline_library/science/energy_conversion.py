from scipy.constants import Planck, electron_volt, speed_of_light


def keV_to_Angstrom(energy_keV: float) -> float:
    """
    Converts energy in keV to wavelength in Ångström.

    Parameters
    ----------
    energy_keV : float
        Energy in keV

    Returns
    -------
    float
        Wavelength in Ångström
    """
    energy_joules = energy_keV * 1000 * electron_volt
    wavelength_SI = Planck * speed_of_light / (energy_joules)
    wavelength_Ångström = wavelength_SI * 1e10
    return wavelength_Ångström


def Angstrom_to_keV(wavelength: float) -> float:
    """
    Converts wavelength in Ångström to energy in keV.

    Parameters
    ----------
    wavelength : float
        Wavelength in Ångström

    Returns
    -------
    float
        Energy in keV
    """
    wavelength_SI = wavelength * 1e-10
    energy_joules = Planck * speed_of_light / (wavelength_SI)
    energy_keV = energy_joules / (1000 * electron_volt)
    return energy_keV
