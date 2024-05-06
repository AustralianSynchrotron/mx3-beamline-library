from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .xray_centering import RasterGridCoordinates


class CenteredLoopMotorCoordinates(BaseModel):
    "Position of the MD3 motors corresponding to an aligned loop (mm)"
    alignment_x: float
    alignment_y: float
    alignment_z: float
    sample_x: float
    sample_y: float
    model_config = ConfigDict(extra="forbid")


class OpticalCenteringResults(BaseModel):
    optical_centering_successful: bool
    centered_loop_coordinates: Optional[CenteredLoopMotorCoordinates] = None
    edge_angle: Optional[float] = Field(None, description="edge angle in degrees")
    flat_angle: Optional[float] = Field(None, description="flat angle in degrees")
    edge_grid_motor_coordinates: Optional[RasterGridCoordinates] = Field(
        None, description="Motor coordinates of the edge grid scan"
    )
    flat_grid_motor_coordinates: Optional[RasterGridCoordinates] = Field(
        None, description="Motor coordinates of the flat grid scan"
    )
    model_config = ConfigDict(extra="forbid")


class BeamCenterModel(BaseModel):
    beam_center: Union[tuple[int, int], list[int]]
    zoom_level: int = Field(description="An integer between 1 and 7")


class AutofocusImage(BaseModel):
    autofocus: bool = True
    min: float = -0.2
    max: float = 1.3


class LoopImageProcessing(BaseModel):
    adaptive_constant: float
    block_size: int


class TopCamera(BaseModel):
    """
    We use the top camera to move the loop to the md3 camera field of view.
    x_pixel_target and y_pixel_target are the pixel coordinates that correspond
    to the position where the loop is seen fully by the md3 camera. These
    values are calculated experimentally and must be callibrated every time the top
    camera is moved.
    Similarly, pixels_per_mm_x and pixels_per_mm_y must be callibrated if the
    top camera is moved
    """

    loop_image_processing: LoopImageProcessing = LoopImageProcessing(
        adaptive_constant=6, block_size=49
    )
    pixels_per_mm_x: float = 69.5
    pixels_per_mm_y: float = 45.0
    x_pixel_target: float = 678.0
    y_pixel_target: float = 430.0
    # Regions of interest
    roi_x: list = [400, 1600]
    roi_y: list = [0, 750]


class MD3PixelsPerMillimeter(BaseModel):
    level_1: float = 520.973
    level_2: float = 622.790
    level_3: float = 797.109
    level_4: float = 1040.905
    level_5: float = 5904.201
    level_6: float = 5503.597
    level_7: float = 8502.362


class MD3Camera(BaseModel):
    """
    We use the top camera to move the loop to the md3 camera field of view.
    x_pixel_target and y_pixel_target are the pixel coordinates that correspond
    to the position where the loop is seen fully by the md3 camera. These
    values are calculated experimentally and must be callibrated every time
    the top camera is moved.
    Similarly, pixels_per_mm_x and pixels_per_mm_y must be callibrated
    if the top camera is moved
    """

    loop_image_processing: LoopImageProcessing = LoopImageProcessing(
        adaptive_constant=3, block_size=35
    )
    pixels_per_mm: MD3PixelsPerMillimeter = MD3PixelsPerMillimeter()


class MD3DefaultPositions(BaseModel):
    """
    This corresponds to a focused sample on the MD3, assuming that the sample
    is aligned with the center of the beam
    """

    alignment_x: float = 0.434


class OpticalCenteringExtraConfig(BaseModel):
    """
    This contains configuration of the optical centering plan that
    should not change often.
    """

    optical_centering_percentage_error: float = 7
    autofocus_image: AutofocusImage = AutofocusImage()
    md3_camera: MD3Camera = MD3Camera()
    top_camera: TopCamera = TopCamera()
    motor_default_positions: MD3DefaultPositions = MD3DefaultPositions()
