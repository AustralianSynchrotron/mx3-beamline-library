from os import environ

from ..classes.detectors import DectrisDetector
from .classes.detectors import SimBlackFlyCam

SIMPLON_API = environ.get("SIMPLON_API", "http://0.0.0.0:8000")

dectris_detector = DectrisDetector(REST=SIMPLON_API, name="dectris_detector")

blackfly_camera = SimBlackFlyCam("13SIM1", name="blackfly_camera")
md_camera = SimBlackFlyCam("13SIM2", name="md_camera")
