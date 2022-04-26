""" Beamline detectors """

from .classes.detectors import DectrisDetector

dectris_detector = DectrisDetector(
    REST="http://sim_plon_api:8000", name="dectris_detector"
)
