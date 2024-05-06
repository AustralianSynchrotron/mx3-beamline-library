from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class MotorCoordinates(BaseModel):
    sample_x: float
    sample_y: float
    alignment_x: float
    alignment_y: float
    alignment_z: float
    omega: Optional[float] = None
    plate_translation: Optional[float] = None
    model_config = ConfigDict(extra="forbid")


class CrystalPositions(BaseModel):
    """
    Crystal positions assuming that the shape of the crystal is approximated by
    a rectangle
    """

    bottom_left_pixel_coords: tuple[int, int] = Field(
        description="Bottom left pixel coordinates of the rectangle surrounding the "
        "crystal"
    )
    top_right_pixel_coords: tuple[int, int] = Field(
        description="Bottom right pixel coordinates of the rectangle surrounding the "
        "crystal"
    )
    width: float = Field(description="Width of the crystal in pixels")
    height: float = Field(description="Height of the crystal in pixels")
    min_x: int = Field(
        description="Minimum x value of the rectangle surrounding the "
        "crystal in pixels"
    )
    max_x: int = Field(
        description="Maximum x value of the rectangle surrounding the "
        "crystal in pixels"
    )
    min_y: int = Field(
        description="Minimum y value of the rectangle surrounding the "
        "crystal in pixels"
    )
    max_y: int = Field(
        description="Maximum y value of the rectangle surrounding the "
        "crystal in pixels"
    )
    bottom_left_motor_coordinates: Optional[MotorCoordinates] = Field(
        None,
        description="Bottom left motor coordinates of the rectangle surrounding the "
        "crystal",
    )
    top_right_motor_coordinates: Optional[MotorCoordinates] = Field(
        None,
        description="Top right motor coordinates of the rectangle surrounding the "
        "crystal",
    )
    center_of_mass_pixels: Optional[tuple[int, int]] = Field(
        None, description="Center of mass of the crystal in pixels"
    )
    center_of_mass_motor_coordinates: Optional[MotorCoordinates] = Field(
        None, description="Motor coordinates of the center of mass of a crystal"
    )
    width_micrometers: Optional[float] = Field(
        None, description="Width of the crystal in micrometers"
    )
    height_micrometers: Optional[float] = Field(
        None, description="Height of the crystal in micrometers"
    )
    model_config = ConfigDict(extra="forbid")


class MaximumNumberOfSpots(BaseModel):
    pixel_position: tuple[int, int]
    motor_positions: Optional[MotorCoordinates] = None
    model_config = ConfigDict(extra="forbid")


class CrystalVolume(BaseModel):
    "Crystal Volume model in micrometers"
    width: float = Field(description="Width of the crystal in micrometers")
    height: float = Field(description="Height of the crystal in micrometers")
    depth: float = Field(description="Depth of the crystal in micrometers")
    volume: float = Field(description="Volume of the crystal in micrometers^3")
    model_config = ConfigDict(extra="forbid")


class CrystalFinderResults(BaseModel):
    crystal_locations: list[CrystalPositions] | None = None
    distance_between_crystals: list[dict] | None = None
    maximum_number_of_spots_location: MaximumNumberOfSpots | None = None
