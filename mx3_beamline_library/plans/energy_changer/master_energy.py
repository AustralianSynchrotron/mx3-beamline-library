from mx3_beamline_library.devices.classes import ASBrickMotor
from mx3_beamline_library.plans.energy_changer.dmm_energy import DMMEnergy
from mx3_beamline_library.plans.energy_changer.ivu_energy import IVUEnergy


class MasterEnergy:
    """Master energy class"""

    def __init__(
        self,
        parallel_translation_motor: ASBrickMotor,
        bragg_angle_motor: ASBrickMotor,
        gap_motor: ASBrickMotor,
    ) -> None:
        """
        Parameters
        ----------
        parallel_translation_motor : ASBrickMotor
            The DMM parallel_translation_motor
        bragg_angle_motor : ASBrickMotor
            The DMM bragg_angle_motor
        gap_motor : ASBrickMotor
            The IVU gap_motor

        Returns
        None
        """
        self.dmm_energy = DMMEnergy(
            parallel_translation_motor=parallel_translation_motor,
            bragg_angle_motor=bragg_angle_motor,
        )
        self.ivu_energy = IVUEnergy(gap_motor)

    def set_master_energy(
        self,
        energy: float,
        harmonic: int,
        ivu_energy_offset: float = 0.14,
        bragg_angle_offset: float = -0.0542,
    ):
        """
        This function sets both the DMM energy and IVU energy based on the
        provided master energy and harmonic.

        Parameters
        ----------
        energy : float
            The master energy in keV.
        harmonic : int
            The harmonic number (must be 3 or 5).

        Returns
        -------
        None
        """
        # TODO: read Bragg angle offset from PV
        self.dmm_energy.set_dmm_energy(energy, bragg_angle_offset=bragg_angle_offset)
        self.ivu_energy.set_ivu_energy(
            energy, harmonic, energy_offset=ivu_energy_offset
        )


if __name__ == "__main__":
    from mx3_beamline_library.devices.classes import ASBrickMotor
    from mx3_beamline_library.devices.sim.classes.motors import MX3SimMotor

    # from ophyd import Signal

    parallel_translation_motor = MX3SimMotor(name="parallel_translation_motor")
    bragg_angle_motor = MX3SimMotor(name="bragg_angle_motor")
    gap_motor = MX3SimMotor(name="gap_motor")

    # parallel_translation_motor = ASBrickMotor(prefix='MX3MONO01MOT07', name='dmm_second_parallel_motion')
    # bragg_angle_motor = ASBrickMotor('MX3MONO01MOT03', name="bragg_angle_motor")
    # gap_motor = ASBrickMotor("SR04ID01:BL_GAP_REQUEST", name="gap_motor")
    print(parallel_translation_motor.get())
    print(bragg_angle_motor.get())
    master_energy = MasterEnergy(
        parallel_translation_motor=parallel_translation_motor,
        bragg_angle_motor=bragg_angle_motor,
        gap_motor=gap_motor,
    )
    master_energy.set_master_energy(
        energy=13.0, harmonic=5, ivu_energy_offset=-0.14, bragg_angle_offset=-0.0542
    )
