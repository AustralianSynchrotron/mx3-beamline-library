from mx3_beamline_library.devices.classes import ASBrickMotor
import asyncio

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
            return self._polynomial(
                5.44465629e-3,
                -2.20026623e-1,
                3.36816977,
                -2.24050487e1,
                6.02512260e1,
                energy,
            )
        elif harmonic == 5:
            return self._polynomial(
                6.90950825e-4,
                -4.05438804e-2,
                8.91597297e-1,
                -8.28166789,
                3.21939465e1,
                energy,
            )
        else:
            raise ValueError(
                f"Only the 3th and 5th harmonic are supported, not {harmonic}"
            )

    async def set_ivu_energy(
        self, energy: float, harmonic: int, energy_offset: float
    ) -> None:
        """
        Sets the IVU energy based on the provided energy and harmonic.

        Parameters
        ----------
        energy : float
            The energy in keV.
        harmonic : int
            The harmonic number (must be 3 or 5).
        energy_offset : float
            The IVU energy offset

        Returns
        -------
        None
        """
        gap_setpoint = self._calculate_gap(
            harmonic=harmonic, energy=energy + energy_offset
        )
        self.gap_motor.set(gap_setpoint)
        await self._wait(self.gap_motor)

    async def _wait(self, motor: ASBrickMotor) -> None:
        """Wait for the motor to stop moving.

        Parameters
        ----------
        motor : ASBrickMotor
            An ASBrickMotor instance
        """
        await asyncio.sleep(0.1)
        while motor.moving:
            await asyncio.sleep(0.1)
