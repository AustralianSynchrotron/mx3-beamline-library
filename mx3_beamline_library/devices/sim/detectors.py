from os import environ

from ..classes.detectors import DectrisDetector
from .classes.detectors import BlackFlyCam

_dectris_host = environ.get("DECTRIS_DETECTOR_HOST", "sim_plon_api")
_dectris_port = environ.get("DECTRIS_DETECTOR_PORT", "8000")

dectris_detector = DectrisDetector(
    REST=f"http://{_dectris_host}:{_dectris_port}", name="dectris_detector"
)

blackfly_camera = BlackFlyCam("13SIM1", name="blackfly_camera")
