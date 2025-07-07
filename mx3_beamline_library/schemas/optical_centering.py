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
    # Regions of interest
    roi_x: list = [400, 1600]
    roi_y: list = [0, 750]


class MD3Camera(BaseModel):

    loop_image_processing: LoopImageProcessing = LoopImageProcessing(
        adaptive_constant=3, block_size=35
    )


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
    grid_height_scale_factor: float = 2  # Cut the grid height by this amount


class CC1(BaseModel):
    enable_callbacks: int = 1


class Image(BaseModel):
    nd_array_port: str = "CC1"


class Cam(BaseModel):
    enable_callbacks: int = 1  # True
    array_callbacks: int = 1  # True
    frame_rate_enable: int = 1  # True
    gain_auto: int = 0  # False
    exposure_auto: int = 0  # False

    frame_rate: float = 20.0  # Hz
    gain: float = 1.0
    exposure_time: float = 0.045  # a.k.a acquire time
    acquire_period: float = 0.05
    pixel_format: int = 0  # Mono8


class TopCameraConfig(BaseModel):
    """Top camera PV configuration"""

    cc1: CC1 = CC1()
    image: Image = Image()
    cam: Cam = Cam()
