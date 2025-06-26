from time import sleep

from ophyd import EpicsSignal, EpicsSignalRO

from ...logger import setup_logger

logger = setup_logger()


class Transmission(EpicsSignal):
    """
    Ophyd device used to change the transmission
    """

    def __init__(
        self,
        read_pv,
        is_moving_pv,
        write_pv=None,
        *,
        put_complete=False,
        string=False,
        limits=False,
        name=None,
        **kwargs,
    ):
        super().__init__(
            read_pv,
            write_pv,
            put_complete=put_complete,
            string=string,
            limits=limits,
            name=name,
            **kwargs,
        )

        self.is_moving_signal = EpicsSignalRO(
            is_moving_pv, name="filter_wheel_is_moving"
        )

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
        while self.is_moving_signal.get():
            sleep(0.02)
