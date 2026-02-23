from time import sleep

from ophyd import Component as Cpt, Device, EpicsSignal, EpicsSignalRO

from mx3_beamline_library.logger import setup_logger

logger = setup_logger(__name__)


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
        self.put(value)

        if not 0 <= value <= 1:
            raise ValueError("The transmission has to be a value between 0 and 1")

        if value == self.get():
            logger.info(f"Transmission is already at value: {value}")
            return

        sleep(0.02)
        while self.is_moving_signal.get():
            sleep(0.02)


class BPM(Device):
    control = Cpt(EpicsSignal, "PreDAC0:OutMux", name="control", lazy=True)
    steering_enable = Cpt(EpicsSignal, "PID:Enable", name="steering_enable", lazy=True)
    x_volt = Cpt(
        EpicsSignal,
        "PreDAC0:Ch2_RBV",
        write_pv="PreDAC0:OutCh2",
        name="x_volt",
        lazy=True,
    )
    y_volt = Cpt(
        EpicsSignal,
        "PreDAC0:Ch1_RBV",
        write_pv="PreDAC0:OutCh1",
        name="y_volt",
        lazy=True,
    )
    x = Cpt(
        EpicsSignal,
        "BPM0:PosX_RBV",
        write_pv="PID:SetpointX",
        name="x",
        lazy=True,
    )
    y = Cpt(
        EpicsSignal,
        "BPM0:PosY_RBV",
        write_pv="PID:SetpointY",
        name="y",
        lazy=True,
    )
    flux = Cpt(EpicsSignal, "BPM0:Int_RBV", name="flux", lazy=True)
    beam_off_threshold = Cpt(
        EpicsSignal,
        "BPM0:BeamOffTh_RBV",
        write_pv="BPM0:BeamOffTh",
        name="beam_off_threshold",
        lazy=True,
    )
