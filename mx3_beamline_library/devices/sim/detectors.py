from ...config import SIMPLON_API
from ..classes.detectors import DectrisDetector
from .classes.detectors import SimBlackFlyCam

dectris_detector = DectrisDetector(REST=SIMPLON_API, name="dectris_detector")

blackfly_camera = SimBlackFlyCam("13SIM1", name="blackfly_camera")
md3_camera = SimBlackFlyCam("13SIM2", name="md3_camera")
