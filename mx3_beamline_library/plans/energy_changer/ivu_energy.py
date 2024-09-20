from mx3_beamline_library.devices.classes import ASBrickMotor


class IVUEnergy:
    """In-Vacuum Undulator Energy class"""

    def __init__(self, gap_motor: ASBrickMotor) -> None:
        """
        Parameters
        ----------
        gap_motor : ASBrickMotor
            The IVU gap motor

        Returns
        -------
        None
        """
        self.gap_motor = gap_motor

    def _polynomial(
        self, a: float, b: float, c: float, d: float, e: float, energy: float
    ):
        """
        Evaluates a polynomial of degree 4.

        Parameters
        ----------
        a, b, c, d, e : float
            Coefficients of the polynomial.
        energy : float
            Input value for the polynomial.

        Returns
        -------
        float
            The evaluated polynomial.
        """
        return a * energy**4 + b * energy**3 + c * energy**2 + d * energy + e

    def _calculate_gap(self, harmonic: int, energy: float):
        """Calculates the IVU gap for a given harmonic.

        Parameters
        ----------
        harmonic : int
            The harmonic number (must be 3 or 5).
        energy : float
            The energy in keV.

        Returns
        -------
        float
            The calculated energy gap.

        Raises
        ------
        ValueError
            If the harmonic is not 3 or 5.
        """
        # TODO: add option to choose harmonic automatically
        if harmonic == 3:
            return self._polynomial(5.44e-3, -2.2e-1, 3.37, -2.24e1, 6.03e1, energy)
        elif harmonic == 5:
            return self._polynomial(6.91e-4, -4.05e-2, 8.92e-1, -8.28, 3.22, energy)
        else:
            raise ValueError(
                f"Only the 3th and 5th harmonic are supported, not {harmonic}"
            )

    def set_ivu_energy(self, energy: float, harmonic: int) -> None:
        """
        Sets the IVU energy based on the provided energy and harmonic.

        Parameters
        ----------
        energy : float
            The energy in keV.
        harmonic : int
            The harmonic number (must be 3 or 5).

        Returns
        -------
        None
        """
        gap_setpoint = self._calculate_gap(harmonic=harmonic, energy=energy)
        self.gap_motor.set(gap_setpoint, wait=True)
