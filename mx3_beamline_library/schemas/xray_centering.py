from typing import Optional, Union

from pydantic import BaseModel, Field


class TestrigEventData(BaseModel):
    dectris_detector_sequence_id: Union[int, float]
    testrig_x_user_setpoint: float
    testrig_x: float
    testrig_z_user_setpoint: Optional[float]
    testrig_z: Optional[float]

    class Config:
        extra = "forbid"


class BlueskyEventDoc(BaseModel):
    descriptor: str
    time: float
    data: Union[TestrigEventData, dict]
    timestamps: Union[TestrigEventData, dict]
    seq_num: int
    uid: str
    filled: dict

    class Config:
        extra = "forbid"


class SpotfinderResults(BaseModel):
    type: str
    number_of_spots: int
    image_id: int
    id: str
    series_id: int
    heatmap_coordinate: Union[tuple[int, int], bytes]
    grid_scan_id: Optional[str] = Field(
        description="Could be either flat or edge for loops, or the drop location for trays. "
        "This parameter is set via the user_data field in the simplon api "
    )

    class Config:
        extra = "forbid"


class RasterGridCoordinates(BaseModel):
    """Raster grid coordinates"""

    use_centring_table: bool = Field(
        description="Determines if the centring table was used during the scan. "
        "If false, then we assume the alignment table was used"
    )
    initial_pos_sample_x: float = Field(
        description="Position of sample x corresponding to the "
        "initial position of the grid (mm)"
    )
    final_pos_sample_x: float = Field(
        description="Position of sample x corresponding to the "
        "final position of the grid (mm)"
    )
    initial_pos_sample_y: float = Field(
        description="Position of sample y corresponding to the "
        "initial position of the grid (mm)"
    )
    final_pos_sample_y: float = Field(
        description="Position of sample y corresponding to the "
        "final position of the grid (mm)"
    )
    initial_pos_alignment_y: float = Field(
        description="Position of alignment y corresponding to the "
        "initial position of the grid (mm)"
    )
    final_pos_alignment_y: float = Field(
        description="Position of alignment x corresponding to the "
        "final position of the grid (mm)."
    )
    initial_pos_alignment_z: Optional[float] = Field(
        description="Position of alignment z corresponding to the "
        "initial position of the grid (mm)"
    )
    final_pos_alignment_z: Optional[float] = Field(
        description="Position of alignment z corresponding to the "
        "final position of the grid (mm)."
    )
    center_pos_sample_x: Optional[float] = Field(
        description="Position of sample_x corresponding to the "
        "center of the grid (x-axis only) (mm). This is only needed for "
        "grid scans where number of columns=1"
    )
    center_pos_sample_y: Optional[float] = Field(
        description="Position of sample_y corresponding to the "
        "center of the grid (x-axis only) (mm). This is only needed for "
        "grid scans where number of columns=1"
    )
    width_mm: float = Field(description="Width of the grid (mm)")
    height_mm: float = Field(description="Height of the grid in (mm)")
    number_of_columns: int
    number_of_rows: int
    omega: float = Field(description="Angle at which the grid scan is done")
    top_left_pixel_coordinates: Optional[tuple[int, int]] = Field(
        description="Top left grid coordinate in units of pixels"
    )
    bottom_right_pixel_coordinates: Optional[tuple[int, int]] = Field(
        description="Bottom right grid coordinate in units of pixels"
    )
    width_pixels: Optional[int] = Field(
        description="Width of the grid in units of pixels"
    )
    height_pixels: Optional[int] = Field(
        description="height of the grid in units of pixels"
    )
    md3_camera_pixel_width: Optional[int] = Field(
        description="Width of the md3 camera in units of pixels"
    )
    md3_camera_pixel_height: Optional[int] = Field(
        description="Height of the md3 camera in units of pixels"
    )
    md3_camera_snapshot: Optional[bytes] = Field(
        description="Snapshot of the md3 camera in byte format obtained using "
        "a combination of the PIL and io libraries"
    )

    class Config:
        extra = "forbid"


class MD3ScanResponse(BaseModel):
    task_name: str
    task_flags: int
    start_time: str
    end_time: str
    task_output: str
    task_exception: str
    result_id: int

    class Config:
        extra = "forbid"
