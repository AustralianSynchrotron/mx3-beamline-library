"""Beamline detectors"""

from ..config import MD3_DB, MD3_HOST, MD3_PORT, SIMPLON_API
from .classes.detectors import BlackFlyCam, DectrisDetector, MDRedisCam

dectris_detector = DectrisDetector(REST=SIMPLON_API, name="dectris_detector")

blackfly_camera = BlackFlyCam("MX3MD3ZOOM0", name="blackfly_camera")

md3_camera = MDRedisCam(r={"host": MD3_HOST, "port": MD3_PORT, "db": MD3_DB})
