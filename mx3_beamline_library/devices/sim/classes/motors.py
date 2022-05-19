""" Simulated motor definitions specific to this beamline. """

import time
from enum import IntEnum

from as_acquisition_library.devices.motors import ASSimMotor
from ophyd import Component as Cpt, Device, MotorBundle
from ophyd.device import DeviceStatus


class MX3SimMotor(ASSimMotor):
    """MX3 Simulated motor"""

    def __init__(
        self,
        *,
        name: str,
        readback_func=None,
        value: float = 0,
        delay: float = 1,
        precision: int = 3,
        parent: Device = None,
        labels: set = None,
        kind: IntEnum = None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        name : str
            Name of the motor
        readback_func : callable, optional
            When the Device is set to ``x``, its readback will be updated to
            ``f(x)``. This can be used to introduce random noise or a systematic
            offset.
            Expected signature: ``f(x) -> value``.
        value : object, optional
            The initial value. Default is 0.
        delay : number, optional
            Simulates how long it takes the device to "move". Default is 1 seconds.
        precision : integer, optional
            Digits of precision. Default is 3.
        parent : Device, optional
            Used internally if this Signal is made part of a larger Device.
        labels : set, optional
            Label of the motor
        kind : a member the Kind IntEnum (or equivalent integer), optional
            Default is Kind.normal. See Kind for options.

        Returns
        -------
        None
        """
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
        """Set the limits of a simulated motor

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
