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
    centered_loop_coordinates: CenteredLoopMotorCoordinates
    edge_angle: float = Field(description="edge angle in degrees")
    flat_angle: float = Field(description="flat angle in degrees")
    edge_grid_motor_coordinates: RasterGridMotorCoordinates = Field(
        description="Motor coordinates of the edge grid scan"
    )
    flat_grid_motor_coordinates: RasterGridMotorCoordinates = Field(
        description="Motor coordinates of the flat grid scan"
    )
