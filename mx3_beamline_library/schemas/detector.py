from typing import Optional

from pydantic import BaseModel, Field, root_validator


class UserData(BaseModel):
    """Data passed to the detector ZMQ-stream"""

    id: str = Field(description="ID of the sample or tray")
    zmq_consumer_mode: str = Field(
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

    @root_validator(pre=True)
    def set_zmq_consumer_mode(cls, values):  # noqa
        allowed_values = ["filewriter", "spotfinder"]
        if values["zmq_consumer_mode"] not in allowed_values:
            raise ValueError(
                "Error setting zmq_consumer_mode. Allowed values are filewriter "
                f"and spotfinder, not {values['zmq_consumer_mode']}"
            )
        return values

    class Config:
        extra = "forbid"


class DetectorConfiguration(BaseModel):
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
