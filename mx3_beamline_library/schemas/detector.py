from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ZMQConsumerMode(Enum):
    SPOTFINDER = "spotfinder"
    FILEWRITER = "filewriter"


class UserData(BaseModel):
    """Data passed to the detector ZMQ-stream"""

    id: str = Field(description="ID of the sample or tray")
    zmq_consumer_mode: str | ZMQConsumerMode = Field(
        default="spotfinder", description="Could be either filewriter or spotfinder"
    )
    number_of_columns: int | None = Field(
        None, description="number of columns of the grid scan"
    )
    number_of_rows: int | None = Field(
        None, description="number of rows of the grid scan"
    )
    grid_scan_id: str | int | None = Field(
        default=None,
        description="Could be either flat or edge for single loops, "
        "or the drop location for trays",
    )
    crystal_id: int | None = None
    data_collection_id: int = 0
    drop_location: str | None = Field(
        None,
        description="The location of the drop used to identify screening datasets",
    )
    model_config = ConfigDict(extra="forbid")


class OmegaModel(BaseModel):
    start: float
    increment: float


class DetectorConfiguration(BaseModel):
    """
    Detector configuration. These keys should match
    the endpoint names of the simplon API, NOT the ZMQ
    stream keys.
    """

    roi_mode: Literal["disabled", "4M"]
    trigger_mode: Literal["eies", "exte", "extg", "exts", "inte", "ints"]
    nimages: int
    frame_time: float
    ntrigger: int
    count_time: Optional[float] = Field(
        None,
        description="The count time should always be less than frame time. "
        "If count time is not set, it will automatically be set to "
        "frame_time - 0.0000001",
    )
    user_data: Optional[UserData] = None
    detector_distance: float
    omega_start: float
    omega_increment: float
    photon_energy: float = Field(
        description="Photon energy in keV. This value is converted internally to "
        "eV since the simplon api expects energy in eV"
    )

    @model_validator(mode="before")
    @classmethod
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

    @model_validator(mode="before")
    @classmethod
    def validate_energy(cls, values):  # noqa
        if values["photon_energy"] / 1000 > 1:
            raise ValueError(
                "The photon energy was most likely specified in eV. "
                "Set the photon energy in keV"
            )
        # convert keV to eV since the simplon api expects eV
        values["photon_energy"] = values["photon_energy"] * 1000
        return values

    model_config = ConfigDict(extra="forbid")
