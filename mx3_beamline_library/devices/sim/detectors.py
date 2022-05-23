from ..classes.detectors import BlackFlyCam, DectrisDetector

dectris_detector = DectrisDetector(
    REST="http://sim_plon_api:8000", name="dectris_detector"
)

blackfly_camera = BlackFlyCam("13SIM1", name="blackfly_camera")
