""" Simulated motor definitions specific to this beamline. """

import time
from enum import IntEnum

import numpy as np
from as_acquisition_library.devices.motors import ASSimMotor
from ophyd import Component as Cpt, Device, MotorBundle, Signal
from ophyd.device import DeviceStatus


class MX3SimMotor(ASSimMotor):
    """MX3 Simulated motor"""

    def __init__(
        self,
        *,
        name: str,
        readback_func=None,
        value: float = 0,
        delay: float = 0.01,
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

    x = Cpt(MX3SimMotor, name="TEST:X")
    y = Cpt(MX3SimMotor, name="TEST:Y")
    z = Cpt(MX3SimMotor, name="TEST:Z")
    phi = Cpt(MX3SimMotor, name="TEST:PHI")


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


class SimMD3Zoom(Signal):
    """
    Ophyd device used to control the zoom level of the MD3
    """

    def __init__(self, name: str, *args, **kwargs) -> None:
        """
        Parameters
        ----------
        motor_name : str
            Motor Name

        Returns
        -------
        None
        """
        super().__init__(name=name, *args, **kwargs)

        self.name = name

    def get(self) -> int:
        """Gets the zoom value

        Returns
        -------
        int
            The zoom value
        """
        return 1

    def _set_and_wait(self, value: float, timeout: float = None) -> None:
        """
        Overridable hook for subclasses to override :meth:`.set` functionality.
        This will be called in a separate thread (`_set_thread`), but will not
        be called in parallel.

        Parameters
        ----------
        value : float
            The value
        timeout : float, optional
            Maximum time to wait for value to be successfully set, or None

        Returns
        -------
        None
        """
        return

    @property
    def position(self) -> int:
        """
        Gets the zoom value.

        Returns
        -------
        int
            The zoom value
        """
        return self.get()

    @property
    def pixels_per_mm(self) -> float:
        """
        Returns the pixels_per_mm value based on the current zoom level of the MD3

        Returns
        -------
        float
            The pixels_per_mm value based on the current zoom level
        """
        return 1500


class SimMD3Phase(Signal):
    def __init__(self, name: str, *args, **kwargs) -> None:
        """
        Parameters
        ----------
        motor_name : str
            Motor Name

        Returns
        -------
        None
        """
        super().__init__(name=name, *args, **kwargs)

        self.name = name

    def get(self) -> str:
        """Gets the current phase

        Returns
        -------
        str
            The current phase
        """
        return "Centring"

    def _set_and_wait(self, value: str, timeout: float = None) -> None:
        """
        Sets the phase of the md3. The allowed values are
        Centring, DataCollection, BeamLocation, and Transfer

        Parameters
        ----------
        value : float
            The value
        timeout : float, optional
            Maximum time to wait for value to be successfully set, or None

        Returns
        -------
        None
        """
        return


class SimMD3BackLight(Signal):
    """
    Ophyd device used to control the phase of the MD3.
    The accepted phases are Centring, DataCollection, BeamLocation, and
    Transfer
    """

    def __init__(self, name: str, *args, **kwargs) -> None:
        """
        Parameters
        ----------
        motor_name : str
            Motor Name

        Returns
        -------
        None
        """
        super().__init__(name=name, *args, **kwargs)

        self.name = name
        self.allowed_values = np.arange(0, 2.1, 0.1)

    def get(self) -> str:
        """Gets the current phase

        Returns
        -------
        str
            The current phase
        """
        return 0.1

    def _set_and_wait(self, value: str, timeout: float = None) -> None:
        """
        Sets the phase of the md3. The allowed values are
        Centring, DataCollection, BeamLocation, and Transfer

        Parameters
        ----------
        value : float
            The value
        timeout : float, optional
            Maximum time to wait for value to be successfully set, or None

        Returns
        -------
        None
        """

        return


class SimMicroDiffractometer(MotorBundle):
    sample_x = Cpt(MX3SimMotor, name="CentringX")
    sample_y = Cpt(MX3SimMotor, name="CentringY")
    alignment_x = Cpt(MX3SimMotor, name="AlignmentX")
    alignment_y = Cpt(MX3SimMotor, name="AlignmentY")
    alignment_z = Cpt(MX3SimMotor, name="AlignmentZ")
    omega = Cpt(MX3SimMotor, name="Omega")
    kappa = Cpt(MX3SimMotor, name="Kappa")
    phi = Cpt(MX3SimMotor, name="Phi")  # This motor is named Kappa phi in mxcube
    aperture_vertical = Cpt(MX3SimMotor, name="ApertureVertical")
    aperture_horizontal = Cpt(MX3SimMotor, name="ApertureHorizontal")
    capillary_vertical = Cpt(MX3SimMotor, name="CapillaryVertical")
    capillary_horizontal = Cpt(MX3SimMotor, name="CapillaryHorizontal")
    scintillator_vertical = Cpt(MX3SimMotor, name="ScintillatorVertical")
    beamstop_x = Cpt(MX3SimMotor, name="BeamstopX")
    beamstop_y = Cpt(MX3SimMotor, name="BeamstopY")
    beamstop_z = Cpt(MX3SimMotor, name="BeamstopZ")
    zoom = Cpt(SimMD3Zoom, name="Zoom")
    phase = Cpt(SimMD3Phase, name="Phase")
    backlight = Cpt(SimMD3BackLight, name="Backlight")
