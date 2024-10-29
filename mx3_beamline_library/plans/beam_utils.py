from urllib.parse import urljoin

from httpx import Client

from ..config import BEAM_CENTER_16M, SIMPLON_API
from ..logger import setup_logger

logger = setup_logger()


def beam_center_16M_to_4M(
    simplon_api: str | None = None, beam_center_16M: tuple[float, float] | None = None
) -> tuple[float, float]:
    """
    Convert beam center coordinates from 16M to 4M mode, and sets the new
    beam center values to the simplon API. The final detector
    roi_mode is set to 4M.

    Parameters
    ----------
    simplon_api : str | None
        Simplon API URL. If None, the default simplon API URL from the
        environment variable is used.
    beam_center_16M : tuple[float,float] | None
        Beam center coordinates in 16M mode. If None, the default beam center
        is set to BEAM_CENTER_16M (set via the environment variable).

    Returns
    -------
    tuple[float,float]
        Beam center coordinates in 4M mode.
    """
    if simplon_api is None:
        simplon_api = SIMPLON_API

    if beam_center_16M is None:
        beam_center_16M = BEAM_CENTER_16M

    with Client() as client:
        response = client.get(
            urljoin(simplon_api, "/detector/api/1.8.0/config/roi_mode")
        )
        response.raise_for_status()
        initial_roi_mode = response.json()["value"]
        if initial_roi_mode == "4M":
            response = client.put(
                urljoin(simplon_api, "/detector/api/1.8.0/config/roi_mode"),
                json={"value": "disabled"},
            )
            response.raise_for_status()

        # Get 16M detector dimensions
        response = client.get(
            urljoin(simplon_api, "detector/api/1.8.0/config/x_pixels_in_detector")
        )
        response.raise_for_status()
        x_pixels_in_detector_16M = response.json()["value"]

        response = client.get(
            urljoin(simplon_api, "detector/api/1.8.0/config/y_pixels_in_detector")
        )
        response.raise_for_status()
        y_pixels_in_detector_16M = response.json()["value"]

        # Get 4M detector dimensions
        response = client.put(
            urljoin(simplon_api, "detector/api/1.8.0/config/roi_mode"),
            json={"value": "4M"},
        )
        response.raise_for_status()

        response = client.get(
            urljoin(simplon_api, "detector/api/1.8.0/config/x_pixels_in_detector")
        )
        response.raise_for_status()
        x_pixels_in_detector_4M = response.json()["value"]

        response = client.get(
            urljoin(simplon_api, "detector/api/1.8.0/config/y_pixels_in_detector")
        )
        response.raise_for_status()
        y_pixels_in_detector_4M = response.json()["value"]

        beam_X_offset = (x_pixels_in_detector_16M - x_pixels_in_detector_4M) / 2
        beam_X = beam_center_16M[0] - beam_X_offset

        beam_Y_offset = (y_pixels_in_detector_16M - y_pixels_in_detector_4M) / 2
        beam_Y = beam_center_16M[1] - beam_Y_offset

        response = client.put(
            urljoin(simplon_api, "/detector/api/1.8.0/config/beam_center_x"),
            json={"value": beam_X},
        )
        response.raise_for_status()

        response = client.put(
            urljoin(simplon_api, "/detector/api/1.8.0/config/beam_center_y"),
            json={"value": beam_Y},
        )
    response.raise_for_status()
    logger.info(f"Beam center set to {(beam_X, beam_Y)}")
    return (beam_X, beam_Y)


def set_beam_center_16M(
    simplon_api: str | None = None, beam_center_16M: tuple[float, float] | None = None
) -> tuple[float, float]:
    if simplon_api is None:
        simplon_api = SIMPLON_API

    if beam_center_16M is None:
        beam_center_16M = BEAM_CENTER_16M

    with Client() as client:
        response = client.put(
            urljoin(simplon_api, "/detector/api/1.8.0/config/beam_center_x"),
            json={"value": beam_center_16M[0]},
        )
        response.raise_for_status()

        response = client.put(
            urljoin(simplon_api, "/detector/api/1.8.0/config/beam_center_y"),
            json={"value": beam_center_16M[1]},
        )
        response.raise_for_status()
    logger.info(f"Beam center set to {beam_center_16M}")
    return beam_center_16M
