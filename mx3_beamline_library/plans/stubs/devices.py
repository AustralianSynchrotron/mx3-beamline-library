from bluesky.utils import Msg, root_ancestor, separate_devices

from ...config import BL_ACTIVE
from ...devices.motors import md3
from ...schemas.optical_centering import RasterGridCoordinates


def stage(devices):
    devices = separate_devices(root_ancestor(device) for device in devices)
    for d in devices:
        yield Msg("stage", d)


def unstage(devices):
    devices = separate_devices(root_ancestor(device) for device in devices)

    for d in reversed(devices):
        yield Msg("unstage", d)


def validate_raster_grid_limits(raster_grid_model: RasterGridCoordinates) -> None:
    """
    Checks if the limits of the raster grid coordinates are valid

    Parameters
    ----------
    raster_grid_model : RasterGridCoordinates
        A RasterGridCoordinates pydantic model

    Returns
    -------
    None
    """
    if BL_ACTIVE == "false":
        return

    # Sample x
    sample_x_limits = md3.sample_x.limits
    _validate_limits(raster_grid_model.initial_pos_sample_x, sample_x_limits)
    _validate_limits(raster_grid_model.final_pos_sample_x, sample_x_limits)

    # Sample_y
    sample_y_limits = md3.sample_y.limits
    _validate_limits(raster_grid_model.initial_pos_sample_y, sample_y_limits)
    _validate_limits(raster_grid_model.final_pos_sample_y, sample_y_limits)

    # Alignment y
    alignment_y_limits = md3.alignment_y.limits
    _validate_limits(raster_grid_model.initial_pos_alignment_y, alignment_y_limits)
    _validate_limits(raster_grid_model.final_pos_alignment_y, alignment_y_limits)

    # Alignment z
    alignment_z_limits = md3.alignment_z.limits
    _validate_limits(raster_grid_model.initial_pos_alignment_z, alignment_z_limits)
    _validate_limits(raster_grid_model.final_pos_alignment_z, alignment_z_limits)


def _validate_limits(value: float, limits: tuple[int, int]) -> None:
    """
    Checks if limits[0]<=value<=limits[1]

    Parameters
    ----------
    value : float
        The value
    limits : tuple[int, int]
        The limits

    Raises
    ------
    ValueError
        Raises an error if the value is not between the limits
    """
    if not limits[0] <= value <= limits[1]:
        raise ValueError(
            f"Value is out of limits. Given value was {value}, "
            f"and the limits are {limits}"
        )
