from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field, root_validator


class ZMQConsumerMode(Enum):
    SPOTFINDER = "spotfinder"
    FILEWRITER = "filewriter"


class UserData(BaseModel):
    """Data passed to the detector ZMQ-stream"""

    id: str = Field(description="ID of the sample or tray")
    zmq_consumer_mode: Union[str, ZMQConsumerMode] = Field(
        default="spotfinder", description="Could be either filewriter or spotfinder"
    )
    number_of_columns: Optional[int] = Field(
        description="number of columns of the grid scan"
    )
    number_of_rows: Optional[int] = Field(description="number of rows of the grid scan")
    grid_scan_id: Optional[str] = Field(
        default=None,
        description="Could be either flat or edge for single loops, "
        "or the drop location for trays",
    )
    crystal_id: Optional[int] = None
    data_collection_id: Optional[int] = 0
    drop_location: Optional[str] = Field(
        description="The location of the drop used to identify screening datasets",
    )

    class Config:
        extra = "forbid"


class OmegaModel(BaseModel):
    start: float
    increment: float


class Goniometer(BaseModel):
    omega: OmegaModel


class DetectorConfiguration(BaseModel):
    """
    Detector configuration. These keys should match
    the endpoint names of the simplon API, NOT the ZMQ
    stream keys.
    """

    roi_mode: str = Field(description="allowed values are disabled and 4M]")
    trigger_mode: str
    nimages: int
    frame_time: float
    ntrigger: int
    count_time: Optional[float] = Field(
        description="The count time should always be less than frame time. "
        "If count time is not set, it will automatically be set to "
        "frame_time - 0.0000001"
    )
    user_data: Optional[UserData]
    detector_distance: float
    goniometer: Union[Goniometer, dict, None]
    photon_energy: float

    @root_validator(pre=True)
    def set_count_time(cls, values):  # noqa
        if values.get("count_time") is None:
            values["count_time"] = values["frame_time"] - 0.0000001
            return values

        if values.get("count_time") > values.get("frame_time"):
            raise ValueError(
                "Count time is greater than frame time. Make sure that "
                "frame_time > count_time"
            )
        return values

    @root_validator(pre=True)
    def set_trigger_mode(cls, values):  # noqa
        allowed_values = ["eies", "exte", "extg", "exts", "inte", "ints"]
        if values["trigger_mode"] not in allowed_values:
            raise ValueError(
                f"Error setting trigger mode. Allowed values are {allowed_values}, "
                f"not {values['trigger_mode']}"
            )
        return values

    class Config:
        extra = "forbid"
