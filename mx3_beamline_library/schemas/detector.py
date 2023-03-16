from typing import Optional, Union

from pydantic import BaseModel, Field, root_validator, validator

class UserData(BaseModel):
    sample_id: str
    zmq_consumer_mode: str = Field(
        default="spotfinder",
        description="Could be either filewriter of spotfinder"
    )
    grid_scan_type: Optional[str] = Field(
        description="Could be either `flat` or `edge` or None"
    )

    @root_validator(pre=True)
    def set_zmq_consumer_mode(cls, values):
        allowed_values = ["filewriter", "spotfinder"]
        if values["zmq_consumer_mode"] not in allowed_values:
            raise ValueError(
                "Error setting zmq_consumer_mode. Allowed values are filewriter "
                f"and spotfinder, not {values['zmq_consumer_mode']}"
            )
        return values
    
    @root_validator(pre=True)
    def set_grid_scan_type(cls, values):
        allowed_values = ["flat", "edge", None]
        if values["grid_scan_type"] not in allowed_values:
            raise ValueError(
                f"Error setting grid_scan_type. Allowed values are flat, edge "
                f"or None, not{values['grid_scan_type']}"
            )
        return values
        
        

    

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
    def set_count_time(cls, values):
        if values.get("count_time") is None:
            values["count_time"] = values["frame_time"] - 0.0000001
            return values

        if values.get("count_time") > values.get("frame_time"):
            raise ValueError(
                "Frame time is greater than count time. Make sure that"
                "frame time is at least ")
        return values
    
    @root_validator(pre=True)
    def set_trigger_mode(cls, values):
        allowed_values = ['eies', 'exte', 'extg', 'exts', 'inte', 'ints']
        if values["trigger_mode"] not in allowed_values:
            raise ValueError(
                f"Error setting trigger mode. Allowed values are {allowed_values}, "
                f"not {values['trigger_mode']}"
            )
        return values
   