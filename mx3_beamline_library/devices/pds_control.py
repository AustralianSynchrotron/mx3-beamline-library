import ast
from os import environ, path

import redis.connection
import yaml
import math
#from redis import StrictRedis
import redis
import json
import pandas as pd
import numpy as np

import pickle

from redis.exceptions import ConnectionError

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
    path.join(path.dirname(__file__), "devices", "classes", "energy_config.yml")
) as config:
    ENERGY_CONFIG = yaml.safe_load(config)

class energy_changer:
    def __init__(self): 
        #for offline testing
        self.energy = 13.0
        #self.stripe = 1   
        self.bragg = 1.59
        self.height = 12.0  
        self.para = 110.16
        self.ivu_dmov = 1

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
        #Remove comment in production
        #self.height = mx3.mx3_pds.dmm.vertical_motion.get()[0]
        #height = self.height
        #print(self.height)
        try:
            if 11.9 <= self.height <= 12.1:
                self.stripe = 0
            elif -0.1 <= self.height <= 0.1:
                self.stripe = 1
            elif -12.1 <= self.height <= -11.9:
                self.stripe = 2
            return(self.stripe)
        except:
            #logger.warning("DMM height is not at a known stripe. Exiting")
            self.stripe = 99
            return(self.stripe)
    

    def get_stripe_offset(self):
        #this_stripe = Stripes[str(dmm_stripe_select.get())][2]
        #print(this_stripe)
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
        #bragg = bragg_cs.get()[0]
        bragg = self.bragg
        this_rev_ri_poly =  eval(redis_client.get('reverse_ri_polynomials'))[str(self.stripe)]
        self.reverse_ri_correction = eval(this_rev_ri_poly)
        return(self.reverse_ri_correction)
    
    def calc_uBragg(self):
        hc=12.398419843320026
        #energy_changer.get_stripe(self)
        energy_changer.get_dmm2d(self)
        energy_changer.get_stripe_offset(self)
        energy_changer.correct_RefractiveIndex(self)
        #bl_uBragg = math.degrees(math.asin((hc/energy/(dmm_2d.get()))))
        bl_uBragg = math.degrees(math.asin((hc/self.energy/energy_changer.get_dmm2d(self))))
        ecal_corrected_uBragg = bl_uBragg + self.this_stripe_offset
        self.ri_corrected_uBragg = ecal_corrected_uBragg + self.ri_correction
        #print(bl_uBragg, ecal_corrected_uBragg, ri_corrected_uBragg)
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
        #pitch_lut_data= pd.read_csv(DMM_PITCH_LUT_FILE, header=None, delimiter=r",", names= ['para', 'pitch'])
        pitch_lut_data=pickle.loads(redis_client.get("pitch2_LUT"))
        pitch_lut_data["closest"] = abs(pitch_lut_data["para"] - self.para)
        two_smallest=pitch_lut_data.nsmallest(2,"closest")
        self.target_pitch2=np.interp(self.para,two_smallest["para"],two_smallest["pitch"])
        #print(self.target_pitch2)
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
        #energy_offset=redis_client.get('IVU_master_energy_offsets')[harmonic][0]
        IVU_master_energy_offset=eval(redis_client.get('IVU_master_energy_offsets'))[str(harmonic)]
        #print(IVU_master_energy_offset)
        energy = self.energy - IVU_master_energy_offset
        #equation=IVU_harmonics[str(harmonic)]
        gap_equation=eval(redis_client.get('IVU_harmonics'))[str(harmonic)]
        this_gap = eval(gap_equation)
        #print(this_gap)
        #Commented out for testing
        # ivu_gap.set(this_gap)
        print(f"setting gap to {this_gap} for energy {self.energy}")
        # energy_changer.wait_for_ivu()
        # ivu_taper.set(0.175)
        # energy_changer.wait_for_ivu()
    
    
    def change_energy(self,energy):
        self.energy=energy
        energy_changer.change_ivu_gap(self)
        energy_changer.calc_uBragg(self)
        #uncomment in production
        #bragg_cs.set(self.ri_corrected_uBragg_)
        energy_changer.calc_parallel(self)
        #second_parallel.set(self.para_motor_target)
        energy_changer.calc_pitch2_lut(self)
        energy_changer.get_pitch2_LUT_offset(self)
        self.this_target_pitch2 = self.target_pitch2 + self.pitch2_LUT_offset
        #dmm_pitch2.set(self.this_target_pitch2)
        #wait_for_ivu()
        #wait_for_para()    
    
    def wait_for_para():
        while True:
            #print("waiting for para")
            if para_dmov.get() == 1:
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
    
    def wait_for_ivu():
        while True:
            if ivu_dmov.get() == 1:
                break

redis_connection.set('foo','bar')
redis_client.set('testvalue','3')
#value = eval(redis_client.get('ri_polynomials'))
#print(f"value is {value['1'][1]}")
#print(value.type())
#print(ENERGY_CONFIG["ri_polynomials"]["0"])
#print(eval(redis_client.get('Stripes'))['2'][2])
x = energy_changer()
x.push_params_to_redis()
x.get_stripe()
x.pitch2_LUT_to_redis()
#x.calc_pitch2_lut() 
x.change_energy(13.0)
#print(x.get_dmm2d())
#print(x.get_dmm2d())
#print(x.calc_parallel())
#print(eval(redis_client.get('ri_polynomials'))['0'])
#print(eval(redis_client.get('IVU_master_energy_offsets'))[str('3')][0])
#harmonic="3"
#print(eval(redis_client.get('IVU_master_energy_offsets'))[str(harmonic)])