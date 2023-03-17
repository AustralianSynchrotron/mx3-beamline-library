from pydantic import BaseModel, Field

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
