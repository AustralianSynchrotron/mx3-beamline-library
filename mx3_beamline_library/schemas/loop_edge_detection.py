import numpy.typing as npt
from pydantic import BaseModel, ConfigDict


class RectangleCoordinates(BaseModel):
    top_left: npt.NDArray
    bottom_right: npt.NDArray
    model_config = ConfigDict(arbitrary_types_allowed=True)


class LoopExtremes(BaseModel):
    top: npt.NDArray
    bottom: npt.NDArray
    right: npt.NDArray
    left: npt.NDArray
    model_config = ConfigDict(arbitrary_types_allowed=True)
