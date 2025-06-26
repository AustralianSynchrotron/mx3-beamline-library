from time import sleep

from ophyd import Signal

from ...logger import setup_logger
from ..beam import filter_wheel_is_moving

logger = setup_logger()


class Transmission(Signal):
    """
    Ophyd device used to change the transmission
    """

    def _set_and_wait(self, value: str, timeout: float = None) -> None:
        """
        Sets the transmission and waits for the filter wheel to stop moving before
        returning

        Parameters
        ----------
        value : float
            The transmission value
        timeout : float, optional
            Maximum time to wait for value to be successfully set, or None

        Returns
        -------
        None
        """
        if not 0 <= value <= 1:
            raise ValueError("The transmission has to be a value between 0 and 1")

        if value == self.get():
            logger.info(f"Transmission is already at value: {value}")
            return

        sleep(0.02)
        while filter_wheel_is_moving.get():
            sleep(0.02)
