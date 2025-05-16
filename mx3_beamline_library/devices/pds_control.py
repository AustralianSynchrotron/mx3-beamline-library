import ast
from os import environ, path

environ["BL_ACTIVE"] = "True"
environ["EPICS_CA_ADDR_LIST"] = "10.244.101.10"


import redis.connection
import yaml
import math
#from redis import StrictRedis
import redis
import json
import pandas as pd
import numpy as np
from mx3_beamline_library.devices.classes import ASBrickMotor
import pickle

from redis.exceptions import ConnectionError



from bluesky import RunEngine

from bluesky.callbacks.best_effort import BestEffortCallback
import matplotlib.pyplot as plt
from mx3_csbs.assembly_tree import MX3
from mx3_csbs.assemblies import MX3PDS_DMM
from mx3_beamline_library.plans.commissioning.commissioning import Scan1D, Scan2D
import mx3_csbs.instances as mx3
from ophyd import EpicsSignal 
from ophyd import EpicsMotor
import ophyd
import h5py
import time
from bluesky.plan_stubs import mv

RE = RunEngine({})
bec = BestEffortCallback()
bec.disable_plots()

RE.subscribe(bec)

#dir(mx3.mx3_pds_dmm)


#from .logger import setup_logger
import logging

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)#, decode_responses=True)
#logger = setup_logger()
logger = logging.getLogger(__name__)
# Determine which mode the beamline library is running on, by default it is run
# in SIM mode
BL_ACTIVE = environ.get("BL_ACTIVE", "false").lower()

# Redis connection
REDIS_HOST = environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(environ.get("REDIS_PORT", "6379"))
REDIS_USERNAME = environ.get("REDIS_USERNAME", None)
REDIS_PASSWORD = environ.get("REDIS_PASSWORD", None)
REDIS_DB = int(environ.get("REDIS_DB", "0"))

# full path to config file
#DMM_PITCH_LUT_FILE = environ.get("/home/thomasc/repos/test_energychanger/devices/classes/Stripe0_para_vrs_pitch_LUT1.csv")
#Local file for debugging
DMM_PITCH_LUT_FILE = "/home/thomasc/repos/test_energychanger/devices/classes/Stripe0_para_vrs_pitch_LUT1.csv"

try:
    redis_connection = redis.StrictRedis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        username=REDIS_USERNAME,
        password=REDIS_PASSWORD,
        db=REDIS_DB,
    )
except ConnectionError:
    logger.warning(
        "A redis connection is not available. Some functionalities may be limited."
    )

with open(
    path.join(path.dirname(__file__),  "classes", "energy_config.yml")
) as config:
    ENERGY_CONFIG = yaml.safe_load(config)

class energy_changer:
    def __init__(self): 
        #for offline testing
        #self.energy = 13.0  
        self.bragg = mx3.mx3_pds_dmm.bragg_cs
        self.pitch = mx3.mx3_pds_dmm.second_pitch_cs
        self.height = mx3.mx3_pds_dmm.vertical_motion
        self.para = mx3.mx3_pds_dmm.second_parallel_motion
        self.para_dmov = EpicsSignal("MX3MONO01MOT07.DMOV",name="para_dmov")
        self.ivu_gap = EpicsSignal("SR04ID01:BL_GAP_REQUEST",name="ivu_gap")
        self.ivu_taper = EpicsSignal("SR04ID01:BL_TAPER_REQUEST",name="ivu_taper")
        self.ivu_dmov = EpicsSignal("SR04MCS01CS01:GAP.DMOV", name="ivu_dmov")

        """Changes PDS and IVU energies based on demand

        Parameters
        ----------
        Reads config and LUT from Redis

        Returns
        -------
        Drives DMM and IVU motors for energy change
        """

    def push_params_to_redis(self):
        for k,v in ENERGY_CONFIG.items():
            v_json = json.dumps(v)
            redis_client.set(k,v_json)

    def get_stripe(self):
        try:
            if 11.9 <= self.height.get()[0] <= 12.1:
                self.stripe = 0
            elif -0.1 <= self.height.get()[0] <= 0.1:
                self.stripe = 1
            elif -12.1 <= self.height.get()[0] <= -11.9:
                self.stripe = 2
            return(self.stripe)
        except:
            logger.error("DMM height is not at a known stripe. Exiting")
            self.stripe = 99
            return(self.stripe)
    

    def get_stripe_offset(self):
        energy_changer.get_stripe(self)
        self.this_stripe_offset = eval(redis_client.get('Stripes'))[str(self.stripe)][2]
        return(self.this_stripe_offset)
    
    def get_dmm2d(self):
        energy_changer.get_stripe(self)
        self.this_2d = 2*eval(redis_client.get('Stripes'))[str(self.stripe)][3]
        return(self.this_2d)
    
    def get_pitch2_LUT_offset(self):
        self.pitch2_LUT_offset = float(redis_client.get('pitch2_LUT_offset'))
        return(self.pitch2_LUT_offset)
    
    def correct_RefractiveIndex(self):
        energy=self.energy
        energy_changer.get_stripe(self)
        this_poly =  eval(redis_client.get('ri_polynomials'))[str(self.stripe)]
        self.ri_correction = eval(this_poly)
        return(self.ri_correction)
    
    def reverse_correct_RefractiveIndex(self):
        #Remove comment in production
        bragg = self.bragg.get()[0]
        this_rev_ri_poly =  eval(redis_client.get('reverse_ri_polynomials'))[str(self.stripe)]
        self.reverse_ri_correction = eval(this_rev_ri_poly)
        return(self.reverse_ri_correction)
    
    def calc_uBragg(self):
        hc=12.398419843320026
        energy_changer.get_dmm2d(self)
        energy_changer.get_stripe_offset(self)
        energy_changer.correct_RefractiveIndex(self)
        bl_uBragg = math.degrees(math.asin((hc/self.energy/energy_changer.get_dmm2d(self))))
        ecal_corrected_uBragg = bl_uBragg + self.this_stripe_offset
        self.ri_corrected_uBragg = ecal_corrected_uBragg + self.ri_correction
        return(self.ri_corrected_uBragg)
    
    #Old pitch2 calcs from polynomial. Depreciated for LUT function
    # def calc_pitch2(self):
    #     if para>221.0:
    #         pitch_poly = Stripe_pitch[str( dmm_stripe_select.get())][2]
    #     else:
    #         pitch_poly = Stripe_pitch[str( dmm_stripe_select.get())][1]
    #     target_pitch2 = eval(pitch_poly)
    #     return(target_pitch2)
    
    def pitch2_LUT_to_redis(self):
        pfile = eval(redis_client.get('pitch2_LUT_files'))[str(self.stripe)]
        pitch_lut_data_df= pd.read_csv(pfile, header=None, delimiter=r",", names= ['para', 'pitch'])
        redis_client.set("pitch2_LUT",pickle.dumps(pitch_lut_data_df))


    def calc_pitch2_lut(self):
        pitch_lut_data=pickle.loads(redis_client.get("pitch2_LUT"))
        pitch_lut_data["closest"] = abs(pitch_lut_data["para"] - self.para.get()[0])
        two_smallest=pitch_lut_data.nsmallest(2,"closest")
        self.target_pitch2=np.interp(self.para.get()[0],two_smallest["para"],two_smallest["pitch"])
        return(self.target_pitch2)
    
    def calc_parallel(self):
        energy_changer.calc_uBragg(self)
        offset = 20.0
        para_target = offset/(2*math.sin(self.ri_corrected_uBragg*math.pi/180))
        self.para_motor_target = para_target - 250
        return(self.para_motor_target)
    
    def change_ivu_gap(self):
        #print(type(self.energy))
        if self.energy > 17.0:
            harmonic = 7
        elif 12.3 <= self.energy <= 17.0:
            harmonic = 5
        else:
            self.energy < 12.3
            harmonic = 3
        IVU_master_energy_offset=eval(redis_client.get('IVU_master_energy_offsets'))[str(harmonic)]
        energy = self.energy - IVU_master_energy_offset
        gap_equation=eval(redis_client.get('IVU_harmonics'))[str(harmonic)]
        this_gap = eval(gap_equation)
        #Commented out for testing
        self.ivu_gap.set(this_gap)
        logger.info(f"setting IVU gap to {round(this_gap,4)} for energy {self.energy}")
        energy_changer.wait_for_ivu(self)
        self.ivu_taper.set(0.175)
        energy_changer.wait_for_ivu(self)
    
    
    def change_energy(self,energy):
        self.energy=energy
        energy_changer.get_stripe(self)
        logger.info(f"setting DMM energy to {self.energy} keV")
        energy_changer.change_ivu_gap(self)
        energy_changer.calc_uBragg(self)
        #uncomment in production
        logger.info(f"setting Bragg to {round(self.ri_corrected_uBragg,4)} degrees")
        self.bragg.set(self.ri_corrected_uBragg)
        energy_changer.calc_parallel(self)
        logger.info(f"setting Parallel to {round(self.para_motor_target,4)} mm")
        self.para.set(self.para_motor_target)
        energy_changer.calc_pitch2_lut(self)
        energy_changer.get_pitch2_LUT_offset(self)
        self.this_target_pitch2 = self.target_pitch2 + self.pitch2_LUT_offset
        logger.info(f"setting Pitch2 to {round(self.this_target_pitch2,4)} mrad")
        self.pitch.set(self.this_target_pitch2)
        energy_changer.wait_for_ivu(self)
        energy_changer.wait_for_para(self)    
    
    def wait_for_para(self):
        while True:
            #print("waiting for para")
            if self.para_dmov.get() == 1:
                break
    
    def wait_for_fw():
        while True:
            #print("waiting for para")
            if filter_outb_dmov.get() == 1:
                break
    
    def wait_for_stripe():
        while True:
            #print("waiting for para")
            if height_dmov.get() == 1:
                break
    
    def wait_for_ivu(self):
        while True:
            if self.ivu_dmov.get() == 1:
                break

redis_connection.set('foo','bar')
redis_client.set('testvalue','3')
#value = eval(redis_client.get('ri_polynomials'))
#print(f"value is {value['1'][1]}")
#print(value.type())
#print(ENERGY_CONFIG["ri_polynomials"]["0"])
#print(eval(redis_client.get('Stripes'))['2'][2])
x = energy_changer()
#x.push_params_to_redis()
#x.get_stripe()
#x.pitch2_LUT_to_redis()
#x.calc_pitch2_lut() 
x.change_energy(13.1)
#test
#dmm_height = EpicsMotor("MX3MONO01MOT01",name="dmm_height")
#dmm_height = mx3.mx3_pds_dmm.vertical_motion
#print(dmm_height.get()[0])
#print(x.get_dmm2d())
#print(x.get_dmm2d())
#print(x.calc_parallel())
#print(eval(redis_client.get('ri_polynomials'))['0'])
#print(eval(redis_client.get('IVU_master_energy_offsets'))[str('3')][0])
#harmonic="3"
#print(eval(redis_client.get('IVU_master_energy_offsets'))[str(harmonic)])