from .energy_conversion import keV_to_Ångström


def dose_rate(
    flux: float,
    beam_width: int,
    beam_height: int,
    energy: float | None,
    wavelength: float | None,
    dose_constant: int = 2000,
) -> float:
    """
    Calculate the dose rate in MGy/s.

    Parameters
    ----------
    flux : float
        Photon flux in photons/s at the sample.
    beam_width : int
        Beam width in micrometers at the sample.
    beam_height : int
        Beam height in micrometers at the sample.
    dose_constant : int
        From Holton 2009.
    energy : float | None
        Energy in keV.
    wavelength : float | None
        Wavelength in Ångström.

    Returns
    -------
    float
        Dose rate in MGy/s.
    """
    if energy is not None and wavelength is not None:
        raise ValueError(
            "Only one of 'energy' or 'wavelength' can be provided, not both."
        )
    if energy is None and wavelength is None:
        raise ValueError("Either 'energy' or 'wavelength' must be provided.")

    if wavelength is None:
        wavelength = keV_to_Ångström(energy)

    flux_density = flux / (beam_width * beam_height)
    Gy_per_second = flux_density / (dose_constant * wavelength**-2)
    MGy_per_second = Gy_per_second / 1e6
    return MGy_per_second
