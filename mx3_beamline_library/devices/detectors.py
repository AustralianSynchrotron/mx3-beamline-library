""" Beamline detectors """
from os import environ

from .classes.detectors import BlackFlyCam, DectrisDetector, MDRedisCam

SIMPLON_API = environ.get("SIMPLON_API", "http://0.0.0.0:8000")

dectris_detector = DectrisDetector(REST=SIMPLON_API, name="dectris_detector")

blackfly_camera = BlackFlyCam("01TR1", name="blackfly_camera")

_md3_host = environ.get("MD3_REDIS_HOST", "localhost")
_md3_port = environ.get("MD3_REDIS_PORT", "8379")
_md3_db = environ.get("MD3_REDIS_DB", "0")

md3_camera = MDRedisCam(r={"host": _md3_host, "port": _md3_port, "db": _md3_db})
