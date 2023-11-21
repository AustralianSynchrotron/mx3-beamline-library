""" Beamline detectors """
from os import environ

from .classes.detectors import BlackFlyCam, DectrisDetector, MDRedisCam

SIMPLON_API = environ.get("SIMPLON_API", "http://0.0.0.0:8000")

dectris_detector = DectrisDetector(REST=SIMPLON_API, name="dectris_detector")

blackfly_camera = BlackFlyCam("01TR1", name="blackfly_camera")

_md_host = environ.get("MD_REDIS_HOST", "localhost")
_md_port = environ.get("MD_REDIS_PORT", "8379")
_md_db = environ.get("MD_REDIS_DB", "0")

md_camera = MDRedisCam(r={"host": _md_host, "port": _md_port, "db": _md_db})
