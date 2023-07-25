from pydantic import BaseModel
import numpy.typing as npt

class RectangleCoordinates(BaseModel):
    top_left: npt.NDArray
    bottom_right: npt.NDArray

    class Config:
        arbitrary_types_allowed = True

class LoopExtremes(BaseModel):
    top:npt.NDArray
    bottom: npt.NDArray
    right:npt.NDArray
    left: npt.NDArray

    class Config:
        arbitrary_types_allowed = True