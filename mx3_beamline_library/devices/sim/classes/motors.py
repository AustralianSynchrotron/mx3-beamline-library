"""Simulated motor definitions specific to this beamline."""

import threading
import time
from enum import IntEnum
from os import environ
from typing import Literal

import numpy as np
from mx_robot_library.schemas.common.path import Path
from mx_robot_library.schemas.common.position import Position
from mx_robot_library.schemas.common.tool import Tool
from mx_robot_library.schemas.responses.state import StateResponse
from ophyd import Component as Cpt, Device, MotorBundle, Signal
from ophyd.device import DeviceStatus
from ophyd.sim import SynAxis

SIM_MOTORS_DELAY = float(environ.get("SIM_MOTORS_DELAY", "0.01"))


class MX3SimMotor(SynAxis):
    """MX3 Simulated motor"""

    def __init__(
        self,
        *,
        name: str,
        readback_func=None,
        value: float = 0,
        delay: float = SIM_MOTORS_DELAY,
        precision: int = 3,
        parent: Device = None,
        labels: set = None,
        kind: IntEnum = None,
        **kwargs,
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
            **kwargs,
        )
        self._limits = (-1000.0, 1000.0)
        self._time = time.time()
        self.delay = delay

    def set(self, value: float, wait=True) -> DeviceStatus:
        """Sets the value of a simulated motor

        Parameters
        ----------
        value : float
            The value of the simulated motor
        wait : bool, optional
            If wait=True, we wait until the end of move, by default True

        Returns
        -------
        DeviceStatus
            The status of the device
        """
        self._time = time.time()

        old_setpoint = self.sim_state["setpoint"]
        self.sim_state["setpoint"] = value
        self.sim_state["setpoint_ts"] = time.time()
        self.setpoint._run_subs(
            sub_type=self.setpoint.SUB_VALUE,
            old_value=old_setpoint,
            value=self.sim_state["setpoint"],
            timestamp=self.sim_state["setpoint_ts"],
        )

        def update_state():
            old_readback = self.sim_state["readback"]
            self.sim_state["readback"] = self._readback_func(value)
            self.sim_state["readback_ts"] = time.time()
            self.readback._run_subs(
                sub_type=self.readback.SUB_VALUE,
                old_value=old_readback,
                value=self.sim_state["readback"],
                timestamp=self.sim_state["readback_ts"],
            )
            self._run_subs(
                sub_type=self.SUB_READBACK,
                old_value=old_readback,
                value=self.sim_state["readback"],
                timestamp=self.sim_state["readback_ts"],
            )

        st = DeviceStatus(device=self)
        if wait:
            if self.delay:

                def sleep_and_finish():
                    time.sleep(self.delay)
                    update_state()
                    st.set_finished()

                threading.Thread(target=sleep_and_finish, daemon=True).start()
            else:
                update_state()
                st.set_finished()
        else:
            update_state()
            st.set_finished()
            self.moving = True
            self._time = time.time()

        return st

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
            self._moving = True
        else:
            self._moving = False
        return self._moving

    @moving.setter
    def moving(self, value: bool) -> None:
        """Sets moving status

        Parameters
        ----------
        value : bool
            The new id
        """
        self._moving = value

    @property
    def position(self):
        return self.get().readback

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
        return self.set(position, wait=wait)

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

    @property
    def state(self):
        return "Ready"


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


class SimMovePlateToShelf(Signal):
    """
    Ophyd device used to move a plate to a drop location based on
    (row, column, drop)
    """

    def __init__(self, name: str, *args, **kwargs) -> None:
        """
        Parameters
        ----------
        motor_name : str
            Motor Name
        server : ClientFactory
            A client Factory object

        Returns
        -------
        None
        """
        super().__init__(name=name, *args, **kwargs)

        self.name = name
        self.position = "A1-1"

    def get(self) -> tuple[int, int, int]:
        """Gets the current drop location

        Returns
        -------
        tuple[int, int, int]
            The current row, column, and drop location
        """
        return self.position

    def _set_and_wait(self, value: str, timeout: float = None) -> None:
        """
        Moves the plate to the specified drop position (row, columns, drop)

        Parameters
        ----------
        value : str
            The drop location
        timeout : float, optional
            Maximum time to wait for value to be successfully set, or None

        Returns
        -------
        None
        """
        assert (
            len(value) == 4 or len(value) == 5
        ), "The drop location should be a string following a format similar to e.g. A1-1"
        row = ord(value[0].upper()) - 64
        assert 1 <= row <= 8, "Column must be a letter between A and H"

        column = int(self._find_between_string(value, value[0], "-"))
        assert 1 <= column <= 12, "Row must be a number between 1 and 12"

        drop = int(value[-1])
        assert 1 <= drop <= 3, "Drop must be a number between 1 and 3"

        # Count internally from 0, not 1
        row = row - 1
        column = column - 1
        drop = drop - 1

        self.position = value

    def _find_between_string(self, s: str, first: str, last: str) -> str:
        """
        Finds the string between `first` and `last`

        Parameters
        ----------
        s : str
            A string
        first : str
            The start value
        last : str
            The last value

        Returns
        -------
        str
            The string between first and last
        """
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]


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
    plate_translation = Cpt(MX3SimMotor, name="PlateTranslation")
    move_plate_to_shelf = Cpt(SimMovePlateToShelf, name="MovePlateToShelf")

    def save_centring_position(self) -> None:
        return

    def get_head_type(
        self,
    ) -> Literal["SmartMagnet", "MiniKappa", "Plate", "Permanent", "Unknown"]:
        """
        Gets the type of the MD3 head

        Returns
        -------
        Literal["SmartMagnet", "MiniKappa", "Plate", "Permanent", "Unknown"]
            The type of the MD3 head
        """
        return "SmartMagnet"

    @property
    def state(self) -> str:
        return "Ready"


class SimMotorState(Signal):
    def get(self):
        return StateResponse(
            cmd="state",
            msg="state(0,0,1,PlateGripper,HOME,,0,0,-1,-1,-1,-1,-1,-1,-1,-1,,0,0,100.0,0,0,0.0,72.0,66.0,1,0,0,ERROR 9150 - User acknowledgement required,268435456,431.3,51.4,124.4,0.5,179.8,-1.2,0.3,-19.9,108.3,-0.2,91.5,0.0,Error: Problem Mounting Plate On Gonio,0,,0,0,22.9,42.8,22.3,22.1,0,0.0,0,0,0,0,0,0,changetool|3|3|0|-1.169|-0.16|392.24|0.0|0.0|-2.083)",  # noqa
            error=None,
            raw_values=(
                "0",
                "0",
                "1",
                "PlateGripper",
                "HOME",
                "",
                "0",
                "0",
                "-1",
                "-1",
                "-1",
                "-1",
                "-1",
                "-1",
                "-1",
                "-1",
                "",
                "0",
                "0",
                "100.0",
                "0",
                "0",
                "0.0",
                "72.0",
                "66.0",
                "1",
                "0",
                "0",
                "ERROR 9150 - User acknowledgement required",
                "268435456",
                "431.3",
                "51.4",
                "124.4",
                "0.5",
                "179.8",
                "-1.2",
                "0.3",
                "-19.9",
                "108.3",
                "-0.2",
                "91.5",
                "0.0",
                "Error: Problem Mounting Plate On Gonio",
                "0",
                "",
                "0",
                "0",
                "22.9",
                "42.8",
                "22.3",
                "22.1",
                "0",
                "0.0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "changetool|3|3|0|-1.169|-0.16|392.24|0.0|0.0|-2.083",
            ),  # noqa
            power=False,
            remote_mode=False,
            fault_or_stopped=True,
            tool=Tool(
                id=6, name="PlateGripper", description="Plate gripper", change_time=None
            ),
            position=Position(id=100, name="HOME", description=""),
            path=Path(id=0, name="", description="Undefined"),
            jaw_a_is_open=False,
            jaw_b_is_open=False,
            jaw_a_pin=None,
            jaw_b_pin=None,
            goni_pin=None,
            arm_plate=None,
            goni_plate=None,
            seq_running=False,
            seq_paused=False,
            speed_ratio=100.0,
            plc_last_msg="ERROR 9150 - User acknowledgement required",
        )


class IsaraRobot(Device):
    # TODO: Properly implement this sim device. This is a quick fix for the
    # queueserver in sim mode
    mount = Cpt(MX3SimMotor, name="mount")
    unmount = Cpt(MX3SimMotor, name="unmount")
    state = Cpt(SimMotorState, name="state")
