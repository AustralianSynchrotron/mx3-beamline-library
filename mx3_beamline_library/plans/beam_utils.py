from urllib.parse import urljoin

from httpx import Client

from ..config import BEAM_CENTER_16M, SIMPLON_API
from ..logger import setup_logger

logger = setup_logger()


def set_beam_center_16M(
    simplon_api: str | None = None, beam_center_16M: tuple[float, float] | None = None
) -> tuple[float, float]:
    if simplon_api is None:
        simplon_api = SIMPLON_API

    if beam_center_16M is None:
        beam_center_16M = BEAM_CENTER_16M

    with Client() as client:
        # Set beam center in 16M mode
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
    logger.info(f"Beam center set to {beam_center_16M}")
    return beam_center_16M