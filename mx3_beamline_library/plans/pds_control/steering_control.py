import time
from typing import Literal

from ophyd import EpicsSignal

from mx3_beamline_library.config import BL_ACTIVE
from mx3_beamline_library.devices.beam import (
    filter_wheel_is_moving,
    kill_goni_lateral,
    kill_goni_vertical,
    transmission,
)
from mx3_beamline_library.devices.classes.beam import BPM
from mx3_beamline_library.devices.motors import md3
from mx3_beamline_library.logger import setup_logger

logger = setup_logger()

ROOT = "MX3DAQIOC04:"
PIDROOT = ROOT + "PID:"
PID_SETTINGS = {
    "KpX": 25.0,
    "KiX": 5.0,
    "KdX": 1.0,
    "eminX": 0.0,
    "ImaxX": 1000,
    "OminX": -1.0,
    "OmaxX": 5.0,
    "OgainX": -1.0,
    "OffsetX": 0.0,
    "FreqX": 4.0,
    "KpY": 1.0,
    "KiY": 0.25,
    "KdY": 0.0,
    "eminY": 0.0,
    "ImaxY": 10,
    "OminY": 1.0,
    "OmaxY": 4.0,
    "OgainY": 1.0,
    "OffsetY": 0.0,
    "FreqY": 4.0,
}


class SteeringControl:
    def __init__(self, bpm: BPM):
        self.ssa_scan_gap = 0.1
        self.bpm = bpm
        if BL_ACTIVE == "true":
            # Force set all steering PVs from PID_SETTINGS dict
            for k, v in PID_SETTINGS.items():
                pvname = PIDROOT + k
                setter = EpicsSignal(pvname, name="setter")
                setter.set(v)
        else:
            logger.warning("BL_ACTIVE is false, not setting steering PVs")

    def toggle_fast_shutter(self, mode: Literal["open", "close"]):
        if mode == "close":
            md3.fast_shutter.set(0)
        elif mode == "open":
            md3.fast_shutter.set(1)
            time.sleep(2)
        else:
            raise ValueError("Fast shutter mode must be 'open' or 'close'")

    def retract_scintillator(self):
        md3.scintillator_vertical.set(-2)

    def insert_scintillator(self):
        md3.scintillator_vertical.set(-0.2199)

    def set_filter_steering(self):
        transmission.set(0.05)
        time.sleep(1)
        self.wait_for_filter_wheel()

    def set_filter_imaging(self):
        transmission.set(0.00015)
        time.sleep(1)
        self.wait_for_filter_wheel()

    def wait_for_filter_wheel(self):
        while True:
            if filter_wheel_is_moving.get() == 0:
                break
            time.sleep(0.1)

    def show_beam(self):
        self.bpm.steering_enable.set("OFF")
        time.sleep(1)
        self.set_filter_imaging()
        self.toggle_fast_shutter("open")

    def return_to_steering(self):
        self.toggle_fast_shutter("close")
        self.set_filter_steering()
        self.bpm.steering_enable.set("ON")

    def disable_steering(self):
        self.bpm.steering_enable.set("OFF")
        time.sleep(1)
        self.bpm.control.set(0)

    def set_piezo_midpoint(self):
        self.bpm.y_volt.set(2.5)
        self.bpm.x_volt.set(2.0)
        time.sleep(0.5)

    def set_current_pos_and_steer(self):
        kill_goni_lateral.set(1)
        kill_goni_vertical.set(1)
        self.toggle_fast_shutter("close")
        time.sleep(0.5)
        self.set_filter_steering()
        time.sleep(5)
        x_now = round(self.bpm.x.get(), 4)
        y_now = round(self.bpm.y.get(), 4)
        self.bpm.x.set(x_now)
        time.sleep(0.1)
        self.bpm.y.set(y_now)
        time.sleep(0.1)
        flux_now = self.bpm.flux.get()
        new_threshold = flux_now * 0.9
        self.bpm.beam_off_threshold.set(new_threshold)
        self.bpm.control.set(1)
        time.sleep(1)
        self.bpm.steering_enable.set("ON")
        self.bpm.x.set(x_now)
        time.sleep(0.1)
        self.bpm.y.set(y_now)
        time.sleep(0.1)

    def set_epics_control(self):
        self.bpm.steering_enable.set("OFF")
        self.bpm.control.set(0)

    def set_for_staff_alignment(self):
        self.set_epics_control()
        self.set_piezo_midpoint()
        self.set_filter_imaging()
        self.toggle_fast_shutter("open")


if __name__ == "__main__":
    from mx3_beamline_library.devices.beam import bmp_1

    steering_control = SteeringControl(bmp_1)

    steering_control.set_for_staff_alignment()
    # steering_control.run_alignment_loop()
    steering_control.set_current_pos_and_steer()
    steering_control.show_beam()
    steering_control.return_to_steering()
