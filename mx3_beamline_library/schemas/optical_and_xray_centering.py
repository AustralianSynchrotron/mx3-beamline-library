from typing import Optional, Union

from pydantic import BaseModel, Field


class TestrigEventData(BaseModel):
    dectris_detector_sequence_id: Union[int, float]
    testrig_x_user_setpoint: float
    testrig_x: float
    testrig_z_user_setpoint: Optional[float]
    testrig_z: Optional[float]


class BlueskyEventDoc(BaseModel):
    descriptor: str
    time: float
    data: Union[TestrigEventData, dict]
    timestamps: Union[TestrigEventData, dict]
    seq_num: int
    uid: str
    filled: dict


class SpotfinderResults(BaseModel):
    type: str
    number_of_spots: int
    image_id: int
    sample_id: str
    series_id: int
    bluesky_event_doc: Optional[Union[BlueskyEventDoc, dict, bytes]]
    grid_scan_type: Optional[str] = Field(
        description="Could be either `flat` or `edge` "
        "This parameter is set via the user_data field in the simplon api "
    )


class RasterGridMotorCoordinates(BaseModel):
    """Raster grid coordinates measured in units of millimeters"""

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
        "final position of the grid (mm)"
    )
    center_pos_sample_x: float = Field(
        description="Position of sample_x corresponding to the "
        "center of the grid (x-axis only) (mm)"
    )
    center_pos_sample_y: float = Field(
        description="Position of sample_y corresponding to the "
        "center of the grid (x-axis only) (mm)"
    )
    width: float = Field(description="Width of the grid (mm)")
    height: float = Field(description="Height of the grid in (mm)")
    number_of_columns: int
    number_of_rows: int
    omega: float = Field(description="Angle at which the grid scan is done")


class CenteredLoopMotorCoordinates(BaseModel):
    "Position of the MD3 motors corresponding to an aligned loop (mm)"
    alignment_x: float
    alignment_y: float
    alignment_z: float
    sample_x: float
    sample_y: float


class MD3ScanResponse(BaseModel):
    task_name: str
    task_flags: int
    start_time: str
    end_time: str
    task_output: str
    task_exception: str
    result_id: int
