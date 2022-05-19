""" Simulated motor definitions specific to this beamline. """

import time

from as_acquisition_library.devices.motors import ASSimMotor
from ophyd import Component as Cpt, MotorBundle
from ophyd.device import DeviceStatus


class MX3SimMotor(ASSimMotor):
    """MX3 Simulated motor"""

    def __init__(
        self,
        *,
        name,
        readback_func=None,
        value=0,
        delay=1,
        precision=3,
        parent=None,
        labels=None,
        kind=None,
        **kwargs
    ) -> None:
        super().__init__(
            name=name,
            readback_func=readback_func,
            value=value,
            delay=delay,
            precision=precision,
            parent=parent,
            labels=labels,
            kind=kind,
            **kwargs
        )
        self._limits = (-1000.0, 1000.0)
        self._time = time.time()
        self.delay = delay

    @property
    def moving(self) -> bool:
        """
        Returns true if a simulated device is moving. This is determined
        by the delay parameter


        Returns
        -------
        bool
            Returns true if the motor is moving
        """
        if time.time() < self._time + self.delay:
            return True
        return False

    def move(self, position: float, wait: bool = True, **kwargs) -> DeviceStatus:
        """Move motors

        Parameters
        ----------
        position : float
            New position
        wait : bool, optional
            This parameters is not used here, but is kept to keep things
            consistent when we use real devices

        Returns
        -------
        DeviceStatus
            Status of the device
        """
        self._time = time.time()
        return self.set(position)

    @property
    def limits(self) -> tuple[float, float]:
        """
        Gets the motor limits

        Returns
        -------
        tuple[float, float]
            Motor limits
        """
        return self._limits

    @limits.setter
    def limits(self, value: tuple[float, float]) -> None:
        """_summary_

        Parameters
        ----------
        value : tuple[float, float]
            New limits

        Returns
        -------
        None
        """
        self._limits = value


class MySimTable(MotorBundle):
    """A Simulated Generic Table."""

    x = Cpt(MX3SimMotor, name="AXIS:X")
    y = Cpt(MX3SimMotor, name="AXIS:Y")
    z = Cpt(MX3SimMotor, name="AXIS:Z")


class SimulatedPVs(MotorBundle):
    """
    Simulated PVs for Mxcube
    """

    m1 = Cpt(MX3SimMotor, name="MXCUBE:m1")
    m2 = Cpt(MX3SimMotor, name="MXCUBE:m2")
    m3 = Cpt(MX3SimMotor, name="MXCUBE:m3")
    m4 = Cpt(MX3SimMotor, name="MXCUBE:m4")
    m5 = Cpt(MX3SimMotor, name="MXCUBE:m5")
    m6 = Cpt(MX3SimMotor, name="MXCUBE:m6")
    m7 = Cpt(MX3SimMotor, name="MXCUBE:m7")
    m8 = Cpt(MX3SimMotor, name="MXCUBE:m8")
