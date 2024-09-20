import numpy as np
from scipy.constants import Planck, electron_volt, speed_of_light

from mx3_beamline_library.devices.classes import ASBrickMotor


class DMMEnergy:
    """Double Multilayer Monochromator Energy class"""

    def __init__(
        self, parallel_translation_motor: ASBrickMotor, bragg_angle_motor: ASBrickMotor
    ) -> None:
        """
        Parameters
        ----------
        parallel_translation_motor : ASBrickMotor
            The parallel_translation_motor
        bragg_angle_motor : ASBrickMotor
            The bragg_angle_motor

        Returns
        -------
        None
        """
        self.parallel_translation_motor = parallel_translation_motor
        self.bragg_angle_motor = bragg_angle_motor

        self.offset = 20  # mm

    def _energy_to_wavelength(self, energy_kev: float) -> float:
        """
        Converts energy in keV to wavelength in Angstrom

        Parameters
        ----------
        energy_kev : float
            The energy in keV

        Returns
        -------
        float
            The wavelength in Angstrom
        """
        return (Planck * speed_of_light) / (energy_kev * (electron_volt * 1000)) * 1e10

    def _set_bragg_angle(self, energy: float) -> float:
        """
        Calculates and sets the Bragg Angle. The Bragg angle is given by

            theta = arcsin(wavelength / 2d)

        where 2d is the lattice spacing which depends on the stripe type

        Parameters
        ----------
        energy : float
            The energy in keV

        Returns
        -------
        float
            The Bragg angle in radians
        """
        # TODO: For now assume we are in the second stripe, so 2d = 40 [A]
        bragg_angle_radians = np.arcsin(self._energy_to_wavelength(energy) / 40)
        bragg_angle_degrees = bragg_angle_radians * 180 / np.pi
        self.bragg_angle_motor.set(bragg_angle_degrees, wait=True)
        return bragg_angle_radians

    def _set_parallel_translation(self, bragg_angle: float) -> None:
        """
        Calculates and sets the parallel translation.
        The parallel translation R is given by

            R = offset / (2*bragg_angle)

        where offset is a constant

        Parameters
        ----------
        bragg_angle : float
            The Bragg angle in radians

        Returns
        -------
        None
        """
        R = self.offset / (2 * np.sin(bragg_angle))
        self.parallel_translation_motor.set(R, wait=True)

    def set_dmm_energy(self, energy: float) -> None:
        """
        Sets the DMM energy by calculating the corresponding Bragg angle
        and parallel translation values

        Parameters
        ----------
        energy : float
            The energy in keV

        Returns
        -------
        None
        """
        bragg_angle = self._set_bragg_angle(energy)
        self._set_parallel_translation(bragg_angle)
