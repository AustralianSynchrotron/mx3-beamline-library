import json
import math
import pickle
from os import path

import numpy as np
import pandas as pd
import yaml
from ophyd import EpicsSignal

from mx3_beamline_library.config import redis_connection
from mx3_beamline_library.devices.classes.motors import ASBrickMotor
from mx3_beamline_library.logger import setup_logger

logger = setup_logger()


with open(
    path.join(path.dirname(__file__), "energy_changer_config", "energy_config.yml")
) as config:
    ENERGY_CONFIG: dict = yaml.safe_load(config)

with open(
    path.join(
        path.dirname(__file__),
        "energy_changer_config",
        "Stripe0_para_vrs_pitch_LUT1.csv",
    )
) as config:
    STRIPE_0_CONFIG: dict = pd.read_csv(
        config, header=None, delimiter=r",", names=["para", "pitch"]
    )

with open(
    path.join(
        path.dirname(__file__),
        "energy_changer_config",
        "Stripe1_para_vrs_pitch_LUT1.csv",
    )
) as config:
    STRIPE_1_CONFIG: dict = pd.read_csv(
        config, header=None, delimiter=r",", names=["para", "pitch"]
    )

with open(
    path.join(
        path.dirname(__file__),
        "energy_changer_config",
        "Stripe2_para_vrs_pitch_LUT1.csv",
    )
) as config:
    STRIPE_2_CONFIG: dict = pd.read_csv(
        config, header=None, delimiter=r",", names=["para", "pitch"]
    )


class EnergyChanger:
    """Changes PDS and IVU energies based on demand"""

    def __init__(
        self,
        bragg: ASBrickMotor,
        pitch: ASBrickMotor,
        height: ASBrickMotor,
        para: ASBrickMotor,
        para_dmov: EpicsSignal,
        ivu_gap: EpicsSignal,
        ivu_taper: EpicsSignal,
        ivu_dmov: EpicsSignal,
    ):
        self.bragg = bragg
        self.pitch = pitch
        self.height = height
        self.para = para
        self.para_dmov = para_dmov
        self.ivu_gap = ivu_gap
        self.ivu_taper = ivu_taper
        self.ivu_dmov = ivu_dmov

        self.get_stripe()

    def push_params_to_redis(self):
        for k, v in ENERGY_CONFIG.items():
            v_json = json.dumps(v)
            redis_connection.set(k, v_json)

    def get_stripe(self):
        try:
            if 11.9 <= self.height.get()[0] <= 12.1:
                self.stripe = 0
            elif -0.1 <= self.height.get()[0] <= 0.1:
                self.stripe = 1
            elif -12.1 <= self.height.get()[0] <= -11.9:
                self.stripe = 2
            return self.stripe
        except Exception as e:
            logger.error(f"DMM height is not at a known stripe. Exiting: {e}")
            self.stripe = 99
            return self.stripe

    def get_stripe_offset(self):
        self.get_stripe()
        self.this_stripe_offset = eval(redis_connection.get("Stripes"))[
            str(self.stripe)
        ][2]
        return self.this_stripe_offset

    def get_dmm2d(self):
        self.get_stripe()
        self.this_2d = 2 * eval(redis_connection.get("Stripes"))[str(self.stripe)][3]
        return self.this_2d

    def get_pitch2_LUT_offset(self):
        self.pitch2_LUT_offset = float(redis_connection.get("pitch2_LUT_offset"))
        return self.pitch2_LUT_offset

    def correct_RefractiveIndex(self):
        energy = self.energy  # noqa
        self.get_stripe()
        this_poly = eval(redis_connection.get("ri_polynomials"))[str(self.stripe)]
        self.ri_correction = eval(this_poly)
        return self.ri_correction

    def reverse_correct_RefractiveIndex(self):
        # Remove comment in production
        self.bragg.get()[0]
        this_rev_ri_poly = eval(redis_connection.get("reverse_ri_polynomials"))[
            str(self.stripe)
        ]
        self.reverse_ri_correction = eval(this_rev_ri_poly)
        return self.reverse_ri_correction

    def calc_uBragg(self):
        hc = 12.398419843320026
        self.get_dmm2d()
        self.get_stripe_offset()
        self.correct_RefractiveIndex()
        bl_uBragg = math.degrees(math.asin((hc / self.energy / self.get_dmm2d())))
        ecal_corrected_uBragg = bl_uBragg + self.this_stripe_offset
        self.ri_corrected_uBragg = ecal_corrected_uBragg + self.ri_correction
        return self.ri_corrected_uBragg

    # Old pitch2 calcs from polynomial. Depreciated for LUT function
    # def calc_pitch2(self):
    #     if para>221.0:
    #         pitch_poly = Stripe_pitch[str( dmm_stripe_select.get())][2]
    #     else:
    #         pitch_poly = Stripe_pitch[str( dmm_stripe_select.get())][1]
    #     target_pitch2 = eval(pitch_poly)
    #     return(target_pitch2)

    def pitch2_LUT_to_redis(self):
        if self.stripe == 0:
            pitch_lut_data_df = STRIPE_0_CONFIG
        elif self.stripe == 1:
            pitch_lut_data_df = STRIPE_1_CONFIG
        elif self.stripe == 2:
            pitch_lut_data_df = STRIPE_2_CONFIG
        redis_connection.set("pitch2_LUT", pickle.dumps(pitch_lut_data_df))

    def calc_pitch2_lut(self):
        pitch_lut_data = pickle.loads(redis_connection.get("pitch2_LUT"))
        pitch_lut_data["closest"] = abs(pitch_lut_data["para"] - self.para.get()[0])
        two_smallest = pitch_lut_data.nsmallest(2, "closest")
        self.target_pitch2 = np.interp(
            self.para.get()[0], two_smallest["para"], two_smallest["pitch"]
        )
        return self.target_pitch2

    def calc_parallel(self):
        self.calc_uBragg()
        offset = 20.0
        para_target = offset / (2 * math.sin(self.ri_corrected_uBragg * math.pi / 180))
        self.para_motor_target = para_target - 250
        return self.para_motor_target

    def change_ivu_gap(self):
        # print(type(self.energy))
        if self.energy > 17.0:
            harmonic = 7
        elif 12.3 <= self.energy <= 17.0:
            harmonic = 5
        else:
            self.energy < 12.3
            harmonic = 3
        IVU_master_energy_offset = eval(
            redis_connection.get("IVU_master_energy_offsets")
        )[str(harmonic)]
        energy = self.energy - IVU_master_energy_offset  # noqa
        gap_equation = eval(redis_connection.get("IVU_harmonics"))[str(harmonic)]
        this_gap = eval(gap_equation)
        # Commented out for testing
        self.ivu_gap.set(this_gap)
        logger.info(f"setting IVU gap to {round(this_gap, 4)} for energy {self.energy}")
        self.wait_for_ivu()
        self.ivu_taper.set(0.175)
        self.wait_for_ivu()

    def change_energy(self, energy):
        self.energy = energy
        self.get_stripe()
        logger.info(f"setting DMM energy to {self.energy} keV")
        self.change_ivu_gap()
        self.calc_uBragg()
        # uncomment in production
        logger.info(f"setting Bragg to {round(self.ri_corrected_uBragg, 4)} degrees")
        self.bragg.set(self.ri_corrected_uBragg)
        self.calc_parallel()
        logger.info(f"setting Parallel to {round(self.para_motor_target, 4)} mm")
        self.para.set(self.para_motor_target)
        self.calc_pitch2_lut()
        self.get_pitch2_LUT_offset()
        self.this_target_pitch2 = self.target_pitch2 + self.pitch2_LUT_offset
        logger.info(f"setting Pitch2 to {round(self.this_target_pitch2, 4)} mrad")
        self.pitch.set(self.this_target_pitch2)
        self.wait_for_ivu()
        self.wait_for_para()

    def wait_for_para(self):
        while True:
            # print("waiting for para")
            if self.para_dmov.get() == 1:
                break

    def wait_for_ivu(self):
        while True:
            if self.ivu_dmov.get() == 1:
                break


if __name__ == "__main__":
    from ophyd import Signal
    from ophyd.sim import motor1

    para_dmov = Signal(name="para_dmov", value=1)
    ivu_dmov = Signal(name="ivu_dmov", value=1)

    x = EnergyChanger(
        bragg=motor1,
        pitch=motor1,
        height=motor1,
        para=motor1,
        para_dmov=para_dmov,
        ivu_gap=motor1,
        ivu_taper=motor1,
        ivu_dmov=ivu_dmov,
    )
    # x.push_params_to_redis()
    # x.get_stripe()
    x.pitch2_LUT_to_redis()
    # x.calc_pitch2_lut()
    x.push_params_to_redis()
    # x.pitch2_LUT_to_redis()
    x.change_energy(13.1)
# test
# dmm_height = EpicsMotor("MX3MONO01MOT01",name="dmm_height")
# dmm_height = mx3.mx3_pds_dmm.vertical_motion
# print(dmm_height.get()[0])
# print(x.get_dmm2d())
# print(x.get_dmm2d())
# print(x.calc_parallel())
# print(eval(redis_connection.get('ri_polynomials'))['0'])
# print(eval(redis_connection.get('IVU_master_energy_offsets'))[str('3')][0])
# harmonic="3"
# print(eval(redis_connection.get('IVU_master_energy_offsets'))[str(harmonic)])
