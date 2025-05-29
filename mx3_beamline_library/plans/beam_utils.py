from urllib.parse import urljoin

from httpx import Client

from ..config import SIMPLON_API, redis_connection
from ..logger import setup_logger

logger = setup_logger()


def set_beam_center(distance: float) -> tuple[float, float]:
    """
    Sets the beam center in the simplon api. The beam center is obtained
    from redis from the keys `beam_center_x_4M` and `beam_center_x_16M`. The
    coefficients are used to calculate the beam center based on the
    distance following the formula:

    beam_center = a + b * distance + c * distance^2

    where distance is measured in millimeters. The beam center is then set in
    the simplon api, where the roi_mode is checked to determine if the
    beam center should be set for 4M or 16M mode

    Parameters
    ----------
    distance : float
        The sample detector distance in millimeters

    Returns
    -------
    tuple[float, float]
        The beam center
    """
    with Client() as client:
        response = client.get(
            urljoin(SIMPLON_API, "/detector/api/1.8.0/config/roi_mode")
        )
        response.raise_for_status()
        if response.json()["value"] == "disabled":
            beam_center_x, beam_center_y = get_beam_center_16M(distance)
        elif response.json()["value"] == "4M":
            beam_center_x, beam_center_y = get_beam_center_4M(distance)

        response = client.put(
            urljoin(SIMPLON_API, "/detector/api/1.8.0/config/beam_center_x"),
            json={"value": beam_center_x},
        )
        response.raise_for_status()

        response = client.put(
            urljoin(SIMPLON_API, "/detector/api/1.8.0/config/beam_center_y"),
            json={"value": beam_center_y},
        )
        response.raise_for_status()
    logger.info(f"16M beam center set to {(beam_center_x, beam_center_y)}")
    return beam_center_x, beam_center_y


def get_beam_center_4M(distance: float) -> tuple[float, float]:
    """
    Gets the beam center for 4M mode from redis. The beam center is calculated using
    the coefficients stored in redis for the keys `beam_center_x_4M` and
    `beam_center_y_4M`. The coefficients are used to calculate the beam center
    based on the distance following the formula:
    beam_center = a + b * distance + c * distance^2

    where distance is measured in millimeters.

    Parameters
    ----------
    distance : float
        The distance in millimeters

    Returns
    -------
    tuple[float, float]
        The beam center coordinates (x, y)
    """
    coefficients_x = redis_connection.hgetall(name="beam_center_x_4M")
    if not coefficients_x:
        raise ValueError("No beam center x coefficients found for 4M mode in Redis.")
    a = float(coefficients_x[b"a"])
    b = float(coefficients_x[b"b"])
    c = float(coefficients_x[b"c"])
    beam_center_x = a + b * distance + c * distance**2

    coefficients_y = redis_connection.hgetall(name="beam_center_y_4M")
    if not coefficients_y:
        raise ValueError("No beam center y coefficients found for 4M mode in Redis.")
    a = float(coefficients_y[b"a"])
    b = float(coefficients_y[b"b"])
    c = float(coefficients_y[b"c"])
    beam_center_y = a + b * distance + c * distance**2
    return beam_center_x, beam_center_y


def get_beam_center_16M(distance: float) -> tuple[float, float]:
    """
    Gets the beam center for 16M mode from redis. The beam center is calculated using
    the coefficients stored in redis for the keys `beam_center_x_16M` and
    `beam_center_y_16M`. The coefficients are used to calculate the beam center
    based on the distance following the formula:
    beam_center = a + b * distance + c * distance^2

    where distance is measured in millimeters.

    Parameters
    ----------
    distance : float
        The distance in millimeters

    Returns
    -------
    tuple[float, float]
        The beam center coordinates (x, y)
    """
    coefficients_x = redis_connection.hgetall(name="beam_center_x_16M")
    if not coefficients_x:
        raise ValueError("No beam center x coefficients found for 16M mode in Redis.")
    a = float(coefficients_x[b"a"])
    b = float(coefficients_x[b"b"])
    c = float(coefficients_x[b"c"])
    beam_center_x = a + b * distance + c * distance**2

    coefficients_y = redis_connection.hgetall(name="beam_center_y_16M")
    if not coefficients_y:
        raise ValueError("No beam center y coefficients found for 16M mode in Redis.")
    a = float(coefficients_y[b"a"])
    b = float(coefficients_y[b"b"])
    c = float(coefficients_y[b"c"])
    beam_center_y = a + b * distance + c * distance**2
    return beam_center_x, beam_center_y
