from typing import Optional

from pydantic import BaseModel, Field

from .xray_centering import RasterGridMotorCoordinates


class CenteredLoopMotorCoordinates(BaseModel):
    "Position of the MD3 motors corresponding to an aligned loop (mm)"
    alignment_x: float
    alignment_y: float
    alignment_z: float
    sample_x: float
    sample_y: float


class OpticalCenteringResults(BaseModel):
    optical_centering_successful: bool
    centered_loop_coordinates: Optional[CenteredLoopMotorCoordinates]
    edge_angle: Optional[float] = Field(description="edge angle in degrees")
    flat_angle: Optional[float] = Field(description="flat angle in degrees")
    edge_grid_motor_coordinates: Optional[RasterGridMotorCoordinates] = Field(
        description="Motor coordinates of the edge grid scan"
    )
    flat_grid_motor_coordinates: Optional[RasterGridMotorCoordinates] = Field(
        description="Motor coordinates of the flat grid scan"
    )
