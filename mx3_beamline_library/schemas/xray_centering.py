from typing import Optional, Self, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SpotfinderResults(BaseModel):
    type: str
    number_of_spots: int
    image_id: int
    id: str
    series_id: int
    heatmap_coordinate: Union[tuple[int, int], bytes]
    grid_scan_id: Optional[str] = Field(
        None,
        description="Could be either flat or edge for loops, or the drop location for trays. "
        "This parameter is set via the user_data field in the simplon api ",
    )
    model_config = ConfigDict(extra="forbid")


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
        None,
        description="Position of alignment z corresponding to the "
        "initial position of the grid (mm)",
    )
    final_pos_alignment_z: Optional[float] = Field(
        None,
        description="Position of alignment z corresponding to the "
        "final position of the grid (mm).",
    )
    alignment_x_pos: Optional[float] = Field(
        None,
        description="Alignment x. This motor position is not changed during "
        "the scan, but can be saved for future reference",
    )
    plate_translation: Optional[float] = Field(
        None,
        description="Plate translation position. This entry is used "
        "for trays only, and is not changed during the scan.",
    )
    omega: float = Field(description="Angle at which the grid scan is done")
    pixels_per_mm: float = Field(
        description="Pixels per mm. This parameter depends on the zoom lever "
        "at which the grid scan is done and is used only by the "
        "CrystalFinder3D algorithm"
    )
    center_pos_sample_x: Optional[float] = Field(
        None,
        description="Position of sample_x corresponding to the "
        "center of the grid (x-axis only) (mm). This is only needed for "
        "grid scans where number of columns=1",
    )
    center_pos_sample_y: Optional[float] = Field(
        None,
        description="Position of sample_y corresponding to the "
        "center of the grid (x-axis only) (mm). This is only needed for "
        "grid scans where number of columns=1",
    )
    width_mm: float = Field(description="Width of the grid (mm)")
    height_mm: float = Field(description="Height of the grid in (mm)")
    number_of_columns: int
    number_of_rows: int
    top_left_pixel_coordinates: Optional[tuple[int, int]] = Field(
        None, description="Top left grid coordinate in units of pixels"
    )
    bottom_right_pixel_coordinates: Optional[tuple[int, int]] = Field(
        None, description="Bottom right grid coordinate in units of pixels"
    )
    width_pixels: Optional[int] = Field(
        None, description="Width of the grid in units of pixels"
    )
    height_pixels: Optional[int] = Field(
        None, description="height of the grid in units of pixels"
    )
    md3_camera_pixel_width: Optional[int] = Field(
        None, description="Width of the md3 camera in units of pixels"
    )
    md3_camera_pixel_height: Optional[int] = Field(
        None, description="Height of the md3 camera in units of pixels"
    )
    md3_camera_snapshot: Optional[bytes] = Field(
        None,
        description="Snapshot of the md3 camera in byte format obtained using "
        "a combination of the PIL and io libraries",
    )
    model_config = ConfigDict(extra="forbid")


class MD3ScanResponse(BaseModel):
    task_name: str
    task_flags: int
    start_time: str
    end_time: str
    task_output: str
    task_exception: str
    result_id: int | str = Field(description="Can be null if the scan fails")
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_task_exception(cls, values: Self) -> Self:  # noqa
        if values.task_exception != "null":
            raise ValueError(f"The Scan failed with error {values.task_exception}")
        return values
