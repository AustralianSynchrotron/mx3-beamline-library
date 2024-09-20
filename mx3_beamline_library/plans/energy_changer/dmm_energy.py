import numpy as np
from scipy.constants import Planck, electron_volt, speed_of_light

from mx3_beamline_library.devices.classes import ASBrickMotor


class DMMEnergy:
    def __init__(
        self, parallel_translation_motor: ASBrickMotor, bragg_angle_motor: ASBrickMotor
    ) -> None:
        self.parallel_translation_motor = parallel_translation_motor
        self.bragg_angle_motor = bragg_angle_motor

        self.offset = 20  # mm

    def _energy_to_wavelength(self, energy_kev) -> float:
        return (Planck * speed_of_light) / (energy_kev * (electron_volt * 1000)) * 1e10

    def _set_bragg_angle(self, energy: float) -> float:
        # TODO: For now assume we are in the second stripe, so 2d = 40 [A]
        bragg_angle_radians = np.arcsin(self._energy_to_wavelength(energy) / 40)
        bragg_angle_degrees = bragg_angle_radians * 180 / np.pi
        self.bragg_angle_motor.set(bragg_angle_degrees, wait=True)
        return bragg_angle_radians

    def _set_parallel_translation(self, bragg_angle: float) -> None:
        R = self.offset / (2 * np.sin(bragg_angle))
        self.parallel_translation_motor.set(R, wait=True)

    def set_dmm_energy(self, energy):
        bragg_angle = self._set_bragg_angle(energy)
        self._set_parallel_translation(bragg_angle)
