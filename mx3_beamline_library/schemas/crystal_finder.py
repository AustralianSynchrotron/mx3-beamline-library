from typing import Optional

from pydantic import BaseModel, Field


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
    bottom_left_motor_coordinates: Optional[dict] = Field(
        description="Bottom left motor coordinates of the rectangle surrounding the "
        "crystal"
    )
    top_right_motor_coordinates: Optional[dict] = Field(
        description="Top right motor coordinates of the rectangle surrounding the "
        "crystal"
    )