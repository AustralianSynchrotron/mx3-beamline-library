from mx3_beamline_library.devices.classes import ASBrickMotor
from mx3_beamline_library.plans.energy_changer.dmm_energy import DMMEnergy
from mx3_beamline_library.plans.energy_changer.ivu_energy import IVUEnergy


class EnergyChanger(DMMEnergy, IVUEnergy):
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
        DMMEnergy.__init__(self, parallel_translation_motor, bragg_angle_motor)
        IVUEnergy.__init__(self, gap_motor)

    def set_master_energy(self, energy, harmonic: int):
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
        self.set_dmm_energy(energy)
        self.set_ivu_energy(energy, harmonic)


if __name__ == "__main__":
    from mx3_beamline_library.devices.sim.classes.motors import MX3SimMotor

    parallel_translation_motor = MX3SimMotor(name="parallel_translation_motor")
    bragg_angle_motor = MX3SimMotor(name="bragg_angle_motor")
    gap_motor = MX3SimMotor(name="gap_motor")

    energy_changer = EnergyChanger(
        parallel_translation_motor=parallel_translation_motor,
        bragg_angle_motor=bragg_angle_motor,
        gap_motor=gap_motor,
    )
    energy_changer.set_master_energy(energy=13, harmonic=3)
