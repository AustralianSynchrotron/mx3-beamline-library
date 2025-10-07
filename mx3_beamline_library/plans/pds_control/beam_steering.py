from ophyd import EpicsSignal
from mx3_beamline_library.devices.motors import md3
from mx3_beamline_library.devices.beam import transmission, flux, filter_wheel_is_moving
from mx3_beamline_library.config import BL_ACTIVE
from mx3_beamline_library.logger import setup_logger
import time
ROOT = "MX3DAQIOC04:"
PIDROOT = ROOT+"PID:"
PID_SETTINGS = { "KpX" : 25.0,
                 "KiX" : 5.0,
                 "KdX" : 1.0,
                 "eminX" : 0.0,
                 "ImaxX" : 1000,
                 "OminX" : -1.0,
                 "OmaxX" : 5.0,
                 "OgainX" : -1.0,
                 "OffsetX" : 0.0,
                 "FreqX" : 4.0,
                 "KpY" : 1.0,
                 "KiY" : 0.25,
                 "KdY" : 0.0,
                 "eminY" : 0.0,
                 "ImaxY" : 10,
                 "OminY" : 1.0,
                 "OmaxY" : 4.0,
                 "OgainY" : 1.0,
                 "OffsetY" : 0.0,
                 "FreqY" : 4.0 }


logger = setup_logger()
class SteeringControl:
    def __init__(
            self, 
            steering_enable: EpicsSignal, 
            kill_goni_lateral: EpicsSignal, 
            kill_goni_vertical: EpicsSignal,
            control: EpicsSignal,
            x_Volt_SP: EpicsSignal,
            y_Volt_SP: EpicsSignal,
            x_RBV: EpicsSignal,
            y_RBV: EpicsSignal,
            x_SP: EpicsSignal,
            y_SP: EpicsSignal,
            beamOffThreshold_SP: EpicsSignal
            ):
        self.ssa_scan_gap=0.1
        self.steering_enable = steering_enable
        self.kill_goni_lateral = kill_goni_lateral
        self.kill_goni_vertical = kill_goni_vertical
        self.control = control
        self.x_Volt_SP = x_Volt_SP
        self.y_Volt_SP = y_Volt_SP
        self.x_RBV = x_RBV
        self.y_RBV = y_RBV
        self.x_SP = x_SP
        self.y_SP = y_SP
        self.beamOffThreshold_SP = beamOffThreshold_SP
        if BL_ACTIVE == "true":
            logger.info("Forcing all steering PVs from PID_SETTINGS dict")
            #Force set all steering PVs from PID_SETTINGS dict
            for k,v in PID_SETTINGS.items():
                pvname = PIDROOT+k
                setter = EpicsSignal(pvname,name="setter")
                setter.set(v)
        
    def toggle_fast_shutter(self, mode: str):
        if mode == "close":
            md3.fast_shutter.set(0)
        elif mode == "open":
            md3.fast_shutter.set(1)
            time.sleep(2)
        else:
            raise ValueError("Mode must be 'open' or 'close'")
    
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
            
            time.sleep(0.2)

    
    def show_beam(self):
        self.steering_enable.set("OFF")
        time.sleep(1)
        self.set_filter_imaging()
        self.toggle_fast_shutter("open")

    def return_to_steering(self):
        self.toggle_fast_shutter("close")
        self.set_filter_steering()
        self.steering_enable.set("ON")
    
    def disable_steering(self):
        self.steering_enable.set("OFF")
        time.sleep(1)
        self.control.set(0)

    def set_piezo_midpoint(self):
        self.y_Volt_SP.set(2.5)
        self.x_Volt_SP.set(2.0)
        time.sleep(0.5)
        
    def set_current_pos_and_steer(self):
        self.kill_goni_lateral.set(1)
        self.kill_goni_vertical.set(1)
        self.toggle_fast_shutter("close")
        time.sleep(0.5)
        self.set_filter_steering()
        time.sleep(5)
        x_now = round(self.x_RBV.get(),4)
        y_now = round(self.y_RBV.get(),4)
        self.x_Volt_SP.set(x_now)
        time.sleep(0.1)
        self.y_SP.set(y_now)
        time.sleep(0.1)
        flux_now = flux.get()
        new_threshold = flux_now*0.9
        self.beamOffThreshold_SP.set(new_threshold)
        self.control.set(1)
        time.sleep(1)
        self.steering_enable.set("ON")
        self.x_SP.set(x_now)
        time.sleep(0.1)
        self.y_SP.set(y_now)
        time.sleep(0.1)

    def set_epics_control(self):
        self.steering_enable.set("OFF")
        self.control.set(0)

    def set_for_staff_alignment(self):
        self.set_epics_control()
        self.set_piezo_midpoint()
        self.set_filter_imaging()
        self.toggle_fast_shutter("open")

if __name__ == "__main__":
    from ophyd import Signal

    steering_enable = Signal(name="steering_enable", value=0)
    kill_goni_lateral = Signal(name="kill_goni_lateral", value=0)
    kill_goni_vertical = Signal(name="kill_goni_vertical", value=0)
    control = Signal(name="control", value=0)
    x_Volt_SP = Signal(name="x_Volt_SP", value=0)
    y_Volt_SP = Signal(name="y_Volt_SP", value=0)
    x_RBV = Signal(name="x_RBV", value=0)
    y_RBV = Signal(name="y_RBV", value=0)
    x_SP = Signal(name="x_SP", value=0)
    y_SP = Signal(name="y_SP", value=0)
    beamOffThreshold_SP = Signal(name="beamOffThreshold_SP", value=0)
    steering_control = SteeringControl(
        steering_enable=steering_enable,
        kill_goni_lateral=kill_goni_lateral,
        kill_goni_vertical=kill_goni_vertical,
        control=control,
        x_Volt_SP=x_Volt_SP,
        y_Volt_SP=y_Volt_SP,
        x_RBV=x_RBV,
        y_RBV=y_RBV,
        x_SP=x_SP,
        y_SP=y_SP,
        beamOffThreshold_SP=beamOffThreshold_SP
    )
    
    steering_control.set_for_staff_alignment()
    #steering_control.run_alignment_loop()
    steering_control.set_current_pos_and_steer()
    steering_control.show_beam()
    steering_control.return_to_steering()
