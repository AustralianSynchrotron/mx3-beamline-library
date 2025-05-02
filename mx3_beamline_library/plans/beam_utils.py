from urllib.parse import urljoin

from httpx import Client

from ..config import BEAM_CENTER_16M, SIMPLON_API
from ..logger import setup_logger

logger = setup_logger()


def set_beam_center_16M(
    simplon_api: str | None = None, beam_center_16M: tuple[float, float] | None = None
) -> tuple[float, float]:
    """
    Sets the beam center in 16M mode. Note that the simplon api
    rescales the beam center when switching between 16M and 4M modes
    automatically, so we ensure that we always set the beam center
    while in 16M mode.

    Parameters
    ----------
    simplon_api : str | None, optional
        The simplon api url, by default None. If None, the environment variable
        is used.
    beam_center_16M : tuple[float, float] | None, optional
        The beam center in 16M mode, by default None. If None,
        the environment variable is used.

    Returns
    -------
    tuple[float, float]
        The beam center in 16M mode
    """
    if simplon_api is None:
        simplon_api = SIMPLON_API

    if beam_center_16M is None:
        beam_center_16M = BEAM_CENTER_16M

    with Client() as client:
        # Set beam center in 16M mode
        response = client.get(
            urljoin(simplon_api, "/detector/api/1.8.0/config/roi_mode")
        )
        response.raise_for_status()
        if response.json()["value"] != "disabled":
            response = client.put(
                urljoin(simplon_api, "/detector/api/1.8.0/config/roi_mode"),
                json={"value": "disabled"},
            )
            response.raise_for_status()

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
    logger.info(f"16M beam center set to {beam_center_16M}")
    return beam_center_16M
