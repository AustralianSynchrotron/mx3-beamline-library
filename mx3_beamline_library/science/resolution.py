from math import atan, sin
from typing import Literal

from mx3_beamline_library.constants import (
    DETECTOR_HEIGHT_4M,
    DETECTOR_HEIGHT_16M,
    DETECTOR_PIXEL_SIZE_X,
    DETECTOR_PIXEL_SIZE_Y,
    DETECTOR_WIDTH_4M,
    DETECTOR_WIDTH_16M,
)
from mx3_beamline_library.plans.beam_utils import (
    get_beam_center_4M,
    get_beam_center_16M,
)
from mx3_beamline_library.science.energy_conversion import keV_to_Angstrom


def _get_radius(
    distance: float,
    roi_mode: Literal["4M", "16M"],
    beam_center: tuple[float, float] | None,
):
    """Get distance from the beam position to the nearest detector edge.

    Parameters
    ----------
    distance : float
        Distance [mm]
    roi_mode : Literal["4M", "16M"]
        The detector roi mode
    beam_center : tuple[float, float] | None
        The beam center. If beam center is None, the beam center is obtained from redis

    Returns
    -------
    float
        Detector radius [mm]
    """

    if beam_center is None:
        if roi_mode == "4M":
            beam_x, beam_y = get_beam_center_4M(distance)
        elif roi_mode == "16M":
            beam_x, beam_y = get_beam_center_16M(distance)
        else:
            raise ValueError(f"roi mode must be 4M or 16M, not {roi_mode}")

    else:
        beam_x, beam_y = beam_center

    if roi_mode == "4M":
        width = DETECTOR_WIDTH_4M
        height = DETECTOR_HEIGHT_4M
    elif roi_mode == "16M":
        width = DETECTOR_WIDTH_16M
        height = DETECTOR_HEIGHT_16M
    else:
        raise ValueError(f"roi mode must be 4M or 16M, not {roi_mode}")

    pixel_x = DETECTOR_PIXEL_SIZE_X * 1000  # mm
    pixel_y = DETECTOR_PIXEL_SIZE_Y * 1000  # mm

    rrx = min(width - beam_x, beam_x) * pixel_x
    rry = min(height - beam_y, beam_y) * pixel_y
    radius = min(rrx, rry)

    return radius


def calculate_resolution(
    distance: float,
    energy: float,
    roi_mode: Literal["4M", "16M"],
    beam_center: tuple[float, float] | None = None,
) -> float:
    """
    Calculate the resolution as function of the detector radius and
    the distance.

    Parameters
    ----------
    distance : float
        Distance from the sample to the detector [mm]
    energy : float
        The energy [kev]
    roi_mode : Literal["4M", "16M"]
        The detector roi mode
    beam_center : tuple[float, float] | None
        The beam center. If beam center is None, the beam center is obtained from redis

    Returns
    -------
    float
        Resolution [Å]
    """
    radius = _get_radius(distance=distance, roi_mode=roi_mode, beam_center=beam_center)
    wavelength = keV_to_Angstrom(energy)
    theta = atan(radius / distance)

    return wavelength / (2 * sin(theta / 2))
