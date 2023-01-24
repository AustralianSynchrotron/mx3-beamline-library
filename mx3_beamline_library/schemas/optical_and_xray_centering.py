from typing import Optional, Union

from pydantic import BaseModel


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
    sequence_id: int
    bluesky_event_doc: Union[BlueskyEventDoc, dict, bytes]


class RasterGridMotorCoordinates(BaseModel):
    """Raster grid coordinates measured in units of millimeters"""
    initial_pos_sample_x: float
    final_pos_sample_x: float
    initial_pos_sample_y: float
    final_pos_sample_y: float
    initial_pos_alignment_y: float
    final_pos_alignment_y: float
    width: float
    height: float
