"""Motor Definitions"""

import logging
from functools import cached_property
from os import environ
from time import perf_counter, sleep
from typing import Literal, Union

import numpy as np
from ophyd import Component as Cpt, EpicsMotor, Kind, MotorBundle, Signal
from ophyd.device import Device, required_for_connection
from ophyd.epics_motor import HomeEnum
from ophyd.positioner import PositionerBase
from ophyd.signal import EpicsSignal, EpicsSignalRO
from ophyd.status import MoveStatus, Status, wait as status_wait
from ophyd.utils import DisconnectedError
from ophyd.utils.epics_pvs import AlarmSeverity, raise_if_disconnected

from ...config import MD3_ADDRESS, MD3_CONFIG, MD3_PORT
from ...schemas.optical_centering import BeamCenterModel
from . import Register
from .md3.ClientFactory import ClientFactory

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)


try:
    from as_acquisition_library.devices.motors import ASEpicsMotor
except ModuleNotFoundError:
    from ophyd import EpicsMotor as ASEpicsMotor

    logging.warning(
        "as_acquisition_library is not installed, ASBrickMotor will inherit from EpicsMotor"
    )


@Register("In Vacuum Rotational Stepper Motor")
@Register("In Vacuum Stepper Motor")
@Register("Stepper Motor")
class ASBrickMotor(ASEpicsMotor):
    """
    Ophyd Device for an AS Power Brick Channel
    Details and docs taken from spreadsheet. Other PVs a
    re available and can be added if deemed important for Ophyd layer.
    """

    adc_in = Cpt(
        EpicsSignalRO,
        ":AdcIn",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="Assigned adc input_n for channels 1 to 8 on PB",
    )

    amp_enabled = Cpt(
        EpicsSignalRO,
        ":AmpEnabled",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="Amp is enabled or not, i.e. Killed",
    )
    brick_fault = Cpt(
        EpicsSignalRO, ":BRICK_FAULT", kind=Kind.omitted, auto_monitor=True, doc=""
    )

    diff_to_ref_pos = Cpt(
        EpicsSignalRO,
        ":DiffToRefPos",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="MainPos - RefPos calculated by ppmac.",
    )

    fault = Cpt(
        EpicsSignalRO,
        ":FAULT",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="Motor overall fault, this includes motor record and controller faults",
    )

    flags_in = Cpt(
        EpicsSignalRO,
        ":FlagsIn",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="Dedicated flags register assigned to axis",
    )

    following_error = Cpt(
        EpicsSignalRO,
        ":FOLL_ERROR",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="axis following error reported by ppmac",
    )

    hard_override = Cpt(
        EpicsSignal,
        ":HardOverride",
        kind=Kind.omitted,
        doc="Hardware epics control strobe",
    )

    home_now = Cpt(
        EpicsSignal,
        ":HOME_NOW.PROC",
        kind=Kind.omitted,
        doc="Start full homing routine for the axes. Action "
        "depends on the axis configuration",
    )

    init = Cpt(
        EpicsSignal,
        ":INIT.PROC",
        kind=Kind.omitted,
        doc="Motor initialisation, initial phasing",
    )

    in_pos = Cpt(
        EpicsSignalRO,
        ":InPos",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="In Position flag",
    )

    max_pos = Cpt(
        EpicsSignalRO,
        ":MaxPos",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="Controller soft (firm) Positive/Forward travel limit position",
    )

    max_speed = Cpt(
        EpicsSignalRO,
        ":MaxSpeed",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="Controller MaxSpeed limit value",
    )

    min_pos = Cpt(
        EpicsSignalRO,
        ":MinPos",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="Controller soft (firm) Negative/Reverse travel limit position",
    )

    pmac_motor_busy = Cpt(
        EpicsSignalRO,
        ":PmacMotorBusy",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="",
    )

    reset_dtr = Cpt(
        EpicsSignal,
        ":RESET_DTR.PROC",
        kind=Kind.omitted,
        doc="Match readback or main to the reference (real) encoder",
    )

    soft_minus_limit = Cpt(
        EpicsSignalRO,
        ":SoftMinusLimit",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="Controller soft -ve limit hit",
    )

    soft_plus_limit = Cpt(
        EpicsSignalRO,
        ":SoftPlusLimit",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="Controller soft +ve limit hit",
    )

    aux_fault = status1 = Cpt(
        EpicsSignalRO,
        ":status1",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="Auxiliary fault, used to interlock form external hardware",
    )

    stop_kill = Cpt(
        EpicsSignal,
        ":STOP_KILL.PROC",
        kind=Kind.omitted,
        doc="Sequence to send Stop then two time  Kill command "
        "in 0.1 and 0.2 seconds interval",
    )

    uninitialised = Cpt(
        EpicsSignalRO,
        ":UNINIT",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="Indicated uninitialised motor based on value of $(MOTOR):PhaseFound",
    )

    unkill = Cpt(
        EpicsSignal,
        ":UNKILL.PROC",
        kind=Kind.omitted,
        doc="Jog stops the motor, which effectively enable the amplifier",
    )


@Register("Coordinate System Virtual Motor")
class VirtualBrickMotor(EpicsMotor):
    fault = Cpt(
        EpicsSignalRO,
        ":FAULT",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="Motor overall fault, this includes motor record and controller faults",
    )


class MxcubeSimulatedPVs(MotorBundle):
    """
    Simulated PVs for Mxcube
    """

    m1 = Cpt(EpicsMotor, ":m1", lazy=True)
    m2 = Cpt(EpicsMotor, ":m2", lazy=True)
    m3 = Cpt(EpicsMotor, ":m3", lazy=True)
    m4 = Cpt(EpicsMotor, ":m4", lazy=True)
    m5 = Cpt(EpicsMotor, ":m5", lazy=True)
    m6 = Cpt(EpicsMotor, ":m6", lazy=True)
    m7 = Cpt(EpicsMotor, ":m7", lazy=True)
    m8 = Cpt(EpicsMotor, ":m8", lazy=True)


class CosylabMotor(Device, PositionerBase):
    """An CosylabMotor motor record, wrapped in a :class:`Positioner`

    Keyword arguments are passed through to the base class, Positioner.
    This classed is based on the EpicsMotor class, but has different attributes
    """

    # position
    user_readback = Cpt(EpicsSignalRO, "_MON", kind="hinted", auto_monitor=True)
    user_setpoint = Cpt(EpicsSignal, "_SP", limits=True, auto_monitor=True)

    # calibration dial <-> user
    user_offset = Cpt(EpicsSignal, "_OFFSET_SP", kind="config", auto_monitor=True)

    # configuration
    velocity = Cpt(EpicsSignal, "_ACVES_MON", kind="config", auto_monitor=True)
    acceleration = Cpt(
        EpicsSignal, "_RAW_MAX_ACC_MON", kind="config", auto_monitor=True
    )
    motor_egu = Cpt(EpicsSignal, "_SP.EGU", kind="config", auto_monitor=True)

    # motor status
    motor_is_moving = Cpt(
        EpicsSignalRO, "_IN_POSITION_STS", kind="omitted", auto_monitor=True
    )
    motor_done_move = Cpt(
        EpicsSignalRO, "_INPOS_STS", kind="omitted", auto_monitor=True
    )
    high_limit_switch = Cpt(
        EpicsSignal, "_HIGH_LIMIT_STS", kind="omitted", auto_monitor=True
    )
    low_limit_switch = Cpt(
        EpicsSignal, "_LOW_LIMIT_STS", kind="omitted", auto_monitor=True
    )
    direction_of_travel = Cpt(
        EpicsSignal, "_DIRECTION", kind="omitted", auto_monitor=True
    )

    # commands
    motor_stop = Cpt(
        EpicsSignal, "_STOP_MOTION_CMD.PROC", kind="omitted", auto_monitor=True
    )
    home_forward = Cpt(
        EpicsSignal, "_RAW_HOME_CMD.PROC", kind="omitted", auto_monitor=True
    )

    # Move mode. 0=PAUSE, 1=GO
    move_mode = Cpt(EpicsSignal, "_MV_MODE_CMD")
    # Trigger move (if paused)
    trigger_move = Cpt(EpicsSignal, "_MV_CMD.PROC")

    # alarm information
    tolerated_alarm = AlarmSeverity.NO_ALARM

    def __init__(
        self,
        prefix: str,
        *,
        read_attrs: list[str] = None,
        configuration_attrs: list[str] = None,
        name: str = None,
        parent=None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        prefix : str
            The record to use
        read_attrs : list[str], optional
            sequence of attribute names. The signals to be read during data acquisition
            (i.e., in read() and describe() calls)
        configuration_attrs : list[str], optional
            sequence of configuration attributes
        name : str, optional
            The name of the device
        parent : instance or None
            The instance of the parent device, if applicable
        settle_time : float, optional
            The amount of time to wait after moves to report status completion
        timeout : float, optional
            The default timeout to use for motion requests, in seconds.

        Returns
        -------
        None
        """
        if read_attrs is None:
            read_attrs = ["user_readback", "user_setpoint"]

        if configuration_attrs is None:
            configuration_attrs = [
                "motor_egu",
            ]

        super().__init__(
            prefix,
            read_attrs=read_attrs,
            configuration_attrs=configuration_attrs,
            name=name,
            parent=parent,
            **kwargs,
        )

        # Make the default alias for the user_readback the name of the
        # motor itself.
        self.user_readback.name = self.name

        self.motor_done_move.subscribe(self._move_changed)
        self.user_readback.subscribe(self._pos_changed)

        self.settle_time = float(environ.get("SETTLE_TIME", "0.2"))

    @property
    @raise_if_disconnected
    def precision(self) -> None:
        """The precision of the readback PV, as reported by EPICS

        Returns
        -------
        None
        """
        return self.user_readback.precision

    @property
    @raise_if_disconnected
    def egu(self) -> None:
        """The engineering units (EGU) for a position

        Returns
        -------
        None
        """
        return self.motor_egu.get()

    @property
    @raise_if_disconnected
    def limits(self) -> None:
        """Motor limits

        Returns
        -------
        None
        """
        return self.user_setpoint.limits

    @property
    @raise_if_disconnected
    def moving(self) -> bool:
        """Whether or not the motor is moving

        Returns
        -------
        bool
            Whether or not the motor is moving
        """
        # We invert bool, as 1 is in position now
        return not bool(self.motor_is_moving.get(use_monitor=False))

    @raise_if_disconnected
    def stop(self, *, success=False) -> None:
        """Stops the motion of the motor

        Returns
        -------
        None
        """
        self.motor_stop.put(1, wait=False)
        super().stop(success=success)

    @raise_if_disconnected
    def move(self, position: float, wait: bool = True, **kwargs) -> MoveStatus:
        """Move to a specified position, optionally waiting for motion to
        complete.

        Parameters
        ----------
        position : float
            Position to move to
        wait : bool
            Wait until motors have finished moving
        moved_cb : callable
            Call this callback when movement has finished. This callback must
            accept one keyword argument: 'obj' which will be set to this
            positioner instance.
        timeout : float, optional
            Maximum time to wait for the motion. If None, the default timeout
            for this positioner is used.

        Returns
        -------
        status : MoveStatus
            The MoveStatus of the motor

        Raises
        ------
        TimeoutError
            When motion takes longer than `timeout`
        ValueError
            On invalid positions
        RuntimeError
            If motion fails other than timing out
        """
        sleep(self.settle_time)

        self._started_moving = False

        status = super().move(position, **kwargs)

        # Change the status from pause to go
        if not self.move_mode.get():
            self.move_mode.put(1, wait=False)

        self.user_setpoint.put(position, wait=False)
        self.trigger_move.put(1, wait=False)

        try:
            if wait:
                status_wait(status)
        except KeyboardInterrupt:
            self.stop()
            raise
        return status

    @property
    @raise_if_disconnected
    def position(self) -> float:
        """The current position of the motor in its engineering units

        Returns
        -------
        float
            The position of the motor
        """
        return self._position

    @raise_if_disconnected
    def set_current_position(self, pos: float) -> None:
        """Configure the motor user position to the given value

        Parameters
        ----------
        pos : float
           Position to set.

        Returns
        -------
        None
        """
        self.user_setpoint.put(pos, wait=True)

    @raise_if_disconnected
    def home(self, direction: HomeEnum, wait=True, **kwargs) -> MoveStatus:
        """Perform the default homing function in the desired direction

        Parameters
        ----------
        direction : HomeEnum
           Direction in which to perform the home search.
        wait : bool
            Wait for MoveStatus to change

        Returns
        -------
        status : MoveStatus
            The motor MoveStatus
        """
        # Only use forward homing as nto an option with pmac
        direction = HomeEnum.forward

        self._started_moving = False
        position = (self.low_limit + self.high_limit) / 2
        status = super().move(position, **kwargs)

        if direction == HomeEnum.forward:
            self.home_forward.put(1, wait=False)
        else:
            self.home_reverse.put(1, wait=False)

        try:
            if wait:
                status_wait(status)
        except KeyboardInterrupt:
            self.stop()
            raise
        return status

    def check_value(self, pos: float) -> None:
        """Check that the position is within the soft limits

        Parameters
        ----------
        pos : float
            The position of the motor

        Returns
        -------
        None
        """
        self.user_setpoint.check_value(pos)

    @required_for_connection
    @user_readback.sub_value
    def _pos_changed(self, timestamp: float = None, value: float = None, **kwargs):
        """Callback from EPICS, indicating a change in position

        Parameters
        ----------
        timestamp : float, optional
            Timestamp, by default None
        value : float, optional
            Motor position, by default None
        """
        self._set_position(value)

    @required_for_connection
    @motor_done_move.sub_value
    def _move_changed(
        self, timestamp: float = None, value: float = None, sub_type=None, **kwargs
    ) -> None:
        """Callback from EPICS, indicating that movement status has changed

        Parameters
        ----------
        timestamp : float, optional
            Timestamp, by default None
        value : float, optional
            Motor value, by default None
        sub_type : optional
            This parameter is currently not used by this method, by default None

        Returns
        -------
        None
        """
        was_moving = self._moving
        self._moving = value != 1

        started = False
        if not self._started_moving:
            started = self._started_moving = not was_moving and self._moving

        if started:
            self._run_subs(
                sub_type=self.SUB_START, timestamp=timestamp, value=value, **kwargs
            )

        if was_moving and not self._moving:
            success = True
            # Check if we are moving towards the low limit switch
            if self.direction_of_travel.get() == 0:
                if self.low_limit_switch.get() == 1:
                    success = False
            # No, we are going to the high limit switch
            else:
                if self.high_limit_switch.get() == 1:
                    success = False

            self._done_moving(success=success, timestamp=timestamp, value=value)

    @property
    def report(self) -> None:
        """Gets the PV Name

        Parameters
        -------
        rep : str
            PV name

        Returns
        -------
        None
        """
        try:
            rep = super().report
        except DisconnectedError:
            # TODO there might be more in this that gets lost
            rep = {"position": "disconnected"}
        rep["pv"] = self.user_readback.pvname
        return rep


class Testrig(MotorBundle):
    """
    Testrig motors
    """

    x = Cpt(CosylabMotor, ":X", lazy=True)
    y = Cpt(CosylabMotor, ":Y", lazy=True)
    z = Cpt(CosylabMotor, ":Z", lazy=True)
    phi = Cpt(CosylabMotor, ":PHI", lazy=True)


class MD3Signal(Signal):
    def __init__(self, name: str, server: ClientFactory, *args, **kwargs) -> None:
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

        self.server = server
        self.motor_name = name

    def wait_ready(self):
        status: str = "Running"
        while status.lower() == "running" or status.lower() == "on":
            status = self.server.getState()


class MD3Motor(Signal):
    """
    Ophyd device used to talk drive MD3 motors via Exporter
    """

    def __init__(self, motor_name: str, server: ClientFactory, *args, **kwargs) -> None:
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
        super().__init__(name=motor_name, *args, **kwargs)

        self.server = server
        self.motor_name = motor_name

    def _set_and_wait(self, value: float, timeout: float = 20, wait=True) -> None:
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
        wait : bool, optional
            Wait until the end of move, by default True

        Returns
        -------
        None
        """
        limits = self.limits
        if not limits[0] <= value <= limits[1]:
            raise ValueError(
                f"Value is out of limits. Given value was {value}, "
                f"and the limits are {limits}"
            )

        initial_position = self.get()
        if timeout is None:
            timeout = 1
        # Make sure the md3 is ready, otherwise the move will not
        # be executed
        self.wait_ready()

        if wait:
            self.server.moveAndWaitEndOfMove(
                motor=self.motor_name,
                initialPos=initial_position,
                position=value,
                useAttr=True,
                n=1,
                goBack=False,
                timeout=timeout,
                backMove=False,
            )
            self.wait_ready
        else:
            self.server.setMotorPosition(self.name, value)

    @cached_property
    def limits(self) -> tuple[float, float]:
        return tuple(self.server.getMotorDynamicLimits(self.name))

    def get(self) -> float:
        """Gets the position of the motors

        Returns
        -------
        float
            The motor value
        """
        return self.server.getMotorPosition(self.motor_name)

    def stop(self, *, success=False):
        pass

    @property
    def moving(self) -> bool:
        """
        Checks if an MD3 motor is moving

        Returns
        -------
        bool
            Wether a motor is moving or not
        """
        status = self.server.getMotorState(self.name).lower()
        if status == "moving" or status == "running" or status == "on":
            return True
        else:
            return False

    @property
    def position(self) -> float:
        """
        Gets the positions of the motors. This method is used for
        consistency with the EpicsMotor class

        Returns
        -------
        float
            The zoom value
        """
        return self.get()

    def move(self, value: float, timeout: float = 20) -> Status:
        """Moves the motors to a different positions

        Parameters
        ----------
        value : float
            The new position
        timeout : float, optional
            Timeout, by default 20

        Returns
        -------
        Status
            A status object
        """
        return self.set(value, timeout=timeout)

    @property
    def state(self) -> str:
        return self.server.getMotorState(self.motor_name)

    def wait_ready(self):
        status: str = "Running"
        while status.lower() == "running" or status.lower() == "on":
            status = self.server.getState()


class MD3Zoom(Signal):
    """
    Ophyd device used to control the zoom level of the MD3
    """

    def __init__(self, name: str, server: ClientFactory, *args, **kwargs) -> None:
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

        self.server = server
        self.name = name

        self._pixels_per_mm = MD3_CONFIG["pixels_per_mm"]

    def get(self) -> int:
        """Gets the zoom value

        Returns
        -------
        int
            The zoom value
        """
        return self.server.getCoaxialCameraZoomValue()

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
        self.server.setCoaxialCameraZoomValue(value)

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
        return self._pixels_per_mm[f"level_{self.position}"]


class MD3Phase(Signal):
    """
    Ophyd device used to control the phase of the MD3.
    The accepted phases are Centring, DataCollection, BeamLocation, and
    Transfer
    """

    def __init__(self, name: str, server: ClientFactory, *args, **kwargs) -> None:
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

        self.server = server
        self.name = name

    def get(self) -> str:
        """Gets the current phase

        Returns
        -------
        str
            The current phase
        """
        return self.server.getCurrentPhase()

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
        if value == self.get():
            logger.info(f"MD3 is already in phase: {value}")
            return

        try:
            logger.info(f"Changing MD3 phase from {self.get()} to: {value}")
            # Check if there is an activity still running and wait
            # until this activity has finished before executing another command
            current_phase = self.get()
            timeout = perf_counter() + 15
            while current_phase == "Unknown":
                sleep(0.2)
                current_phase = self.get()
                if perf_counter() > timeout:
                    raise RuntimeError(
                        "The phase of the MD3 is Unknown, cannot change the"
                        "phase of the MD3. Check the status of the MD3 and try again"
                    )

            self.server.startSetPhase(value)

            # There is not a wait function on the MD3 phase setter, so the following
            # block a waits for the MD3 to change phase
            current_phase = self.get()
            timeout = perf_counter() + 60
            while current_phase == "Unknown":
                sleep(0.2)
                current_phase = self.get()
                if perf_counter() > timeout:
                    raise RuntimeError(
                        "The phase of the MD3 could not be changed after 1 minute. "
                        "Check the status of the MD3 and try again"
                    )

            status = "Running"
            while status == "Running":
                status = self.server.getState()
                sleep(0.1)
            logger.info(f"Phase changed successfully to {self.get()}")

        except Exception:
            logger.info(
                f"Cannot set phase: {value}. Allowed phase values are "
                "Centring, DataCollection, BeamLocation, and Transfer"
            )


class MD3BackLight(Signal):
    """
    Ophyd device used to control the phase of the MD3.
    The accepted phases are Centring, DataCollection, BeamLocation, and
    Transfer
    """

    def __init__(self, name: str, server: ClientFactory, *args, **kwargs) -> None:
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

        self.server = server
        self.name = name
        self.allowed_values = np.arange(0, 2.1, 0.1)

    def get(self) -> float:
        """Gets the current backlight value

        Returns
        -------
        str
            The current phase
        """
        return self.server.getBackLightFactor()

    def _set_and_wait(self, value: float, timeout: float = None) -> None:
        """
        Sets the backlight value

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

        if value in self.allowed_values:
            self.wait_ready()
            self.server.setBackLightFactor(value)
            self.wait_ready()
        else:
            logger.info(f"Allowed values are: {self.allowed_values}, not {value}")

    def wait_ready(self):
        status: str = "Running"
        while status.lower() == "running" or status.lower() == "on":
            status = self.server.getState()
            sleep(0.1)


class MD3FrontLight(MD3BackLight):
    def get(self) -> str:
        """Gets the current phase

        Returns
        -------
        str
            The current phase
        """
        return self.server.getFrontLightFactor()

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
        if value in self.allowed_values:
            self.wait_ready()
            self.server.setFrontLightFactor(value)
            self.wait_ready()
        else:
            logger.info(f"Allowed values are: {self.allowed_values}, not {value}")

    def wait_ready(self):
        status: str = "Running"
        while status.lower() == "running" or status.lower() == "on":
            status = self.server.getState()
            sleep(0.1)


class MD3PLateTranslation(Signal):
    """
    Ophyd device used to control the plate translation
    """

    def __init__(self, name: str, server: ClientFactory, *args, **kwargs) -> None:
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

        self.server = server
        self.name = name

    def get(self) -> float:
        """Gets the plate translation position

        Returns
        -------
        float
            The plate translation position
        """
        return self.server.getPlateTranslationPosition()

    def _set_and_wait(self, value: float, timeout: float = None) -> None:
        """
        Sets the plate translations position

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
        limits = self.limits
        if not limits[0] <= value <= limits[1]:
            raise ValueError(
                f"Value is out of limits. Given value was {value}, "
                f"and the limits are {limits}"
            )
        self.wait_ready()
        self.server.setPlateTranslationPosition(value)
        self.wait_ready()

    def wait_ready(self):
        status: str = "Running"
        while status.lower() == "running" or status.lower() == "on":
            status = self.server.getState()
            sleep(0.1)

    @property
    def position(self) -> float:
        """
        Gets the plate translation position

        Returns
        -------
        float
            The zoom value
        """
        return self.get()

    @cached_property
    def limits(self) -> tuple[float, float]:
        return tuple(self.server.getMotorDynamicLimits(self.name))


class MD3Focus(Signal):
    """
    Ophyd device used to control the focus position
    """

    def __init__(self, name: str, server: ClientFactory, *args, **kwargs) -> None:
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

        self.server = server

    def get(self) -> float:
        """Gets the focus position

        Returns
        -------
        float
            The focus position
        """
        return self.server.CentringTableFocusPosition

    def _set_and_wait(self, value: float, timeout: float = None) -> None:
        """
        Sets the focus position

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
        self.wait_ready()
        self.server.setCentringTableFocusPosition(value)
        self.wait_ready()

    def get_motor_state(self, motor_name: str) -> str:
        """
        Gets the motor state

        Parameters
        ----------
        motor_name : str
            The motor name

        Returns
        -------
        str
            The motor states
        """
        return self.server.getMotorState(motor_name)

    def wait_ready(self):
        """
        Focus depends on both CentringX and CentringY motors to stop moving
        before it is ready

        Returns
        -------
        None
        """
        while True:
            if not self.is_moving("CentringX") and not self.is_moving("CentringY"):
                break
            sleep(0.1)

    def is_moving(self, motor_name) -> bool:
        """
        Checks if focus is moving. This is a combination of both
        CentringX and CentringY motors

        Returns
        -------
        bool
            Whether a motor is moving or not
        """
        status = self.server.getMotorState(motor_name).lower()
        if status == "moving" or status == "running" or status == "on":
            return True
        else:
            return False

    @property
    def position(self) -> float:
        """
        Gets the focus position

        Returns
        -------
        float
            The focus position
        """
        return self.get()


class MD3MovePlateToShelf(Signal):
    """
    Ophyd device used to move a plate to a drop location based on
    (row, column, drop)
    """

    def __init__(self, name: str, server: ClientFactory, *args, **kwargs) -> None:
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

        self.server = server
        self.name = name

    def get(self) -> str:
        """Gets the current drop location.

        Returns
        -------
        tuple[int, int, int]
            The current row, column, and drop location
        """
        drop_location = self.server.getDropLocation()
        row = chr(drop_location[0] + 65)  # Convert number to letter, e.g. 0=A
        column = drop_location[1] + 1  # Count from 1, not 0
        drop = drop_location[2] + 1  # Count from 1, not 0
        return f"{row}{column}-{drop}"

    def _set_and_wait(self, value: str, timeout: float = None) -> None:
        """
        Moves the plate to the specified drop position (row, columns, drop).
        The row, drop and column are specified in the format e.g. A1-1

        Parameters
        ----------
        value : str
            The drop location, e.g. "A1-1"
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

        self.wait_ready()
        self.server.movePlateToShelf(row, column, drop)
        self.wait_ready()

    def wait_ready(self):
        status: str = "Running"
        while status.lower() == "running" or status.lower() == "on":
            status = self.server.getState()
            sleep(0.1)

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


class BeamCenter(MD3Signal):
    def _set_and_wait(
        self, value: Union[dict, BeamCenterModel], timeout, **kwargs
    ) -> None:
        """
        Sets the beam center at a given zoom level

        Parameters
        ----------
        value : Union[dict, BeamCenterModel]
            A BeamCenterModel pydantic model or a dictionary
        timeout : _type_
            Not used in this signal

        Returns
        -------
        None
        """
        if isinstance(value, dict):
            value = BeamCenterModel.model_validate(value)
        self.server.setCoaxialCameraZoomValue(value.zoom_level)
        sleep(0.1)

        MD3_CLIENT.setBeamPositionHorizontal(value.beam_center[0])
        MD3_CLIENT.setBeamPositionVertical(value.beam_center[1])
        self.wait_ready()

    def get(self) -> BeamCenterModel:
        """Gets the beam center at a given zoom level

        Returns
        -------
        BeamCenterModel
            A Teh beam center and current zoom level
        """
        x = MD3_CLIENT.getBeamPositionHorizontal()
        y = MD3_CLIENT.getBeamPositionVertical()
        zoom = self.server.getCoaxialCameraZoomValue()

        return BeamCenterModel(beam_center=(x, y), zoom_level=zoom)


class MD3FastShutter(Signal):
    """
    Ophyd device used to control the MD3 fast shutter
    """

    def __init__(self, name: str, server: ClientFactory, *args, **kwargs) -> None:
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

        self.server = server
        self.name = name

    def get(self) -> str:
        """Gets the fast shutter state

        Returns
        -------
        str
            The fast shutter state
        """
        return self.server.getFastShutterIsOpen()

    def _set_and_wait(self, value: Literal[0, 1], timeout: float = None) -> None:
        """
        Sets the fast shutter state. The allowed values are 0 (closed) or 1 (open)

        Parameters
        ----------
        value : Literal[0, 1]
            The value
        timeout : float, optional
            Maximum time to wait for value to be successfully set, or None

        Returns
        -------
        None
        """
        if value not in [0, 1]:
            raise ValueError(f"The allowed values are 0 or 1. Given value was {value}")
        self.server.setFastShutterIsOpen(value)


MD3_CLIENT = ClientFactory.instantiate(
    type="exporter", args={"address": MD3_ADDRESS, "port": MD3_PORT}
)


class MicroDiffractometer:
    """
    MD3 motors
    """

    sample_x = MD3Motor("CentringX", MD3_CLIENT)
    sample_y = MD3Motor("CentringY", MD3_CLIENT)
    alignment_x = MD3Motor("AlignmentX", MD3_CLIENT)
    alignment_y = MD3Motor("AlignmentY", MD3_CLIENT)
    alignment_z = MD3Motor("AlignmentZ", MD3_CLIENT)
    omega = MD3Motor("Omega", MD3_CLIENT)
    kappa = MD3Motor("Kappa", MD3_CLIENT)
    phi = MD3Motor("Phi", MD3_CLIENT)  # This motor is named Kappa phi in mxcube
    aperture_vertical = MD3Motor("ApertureVertical", MD3_CLIENT)
    aperture_horizontal = MD3Motor("ApertureHorizontal", MD3_CLIENT)
    capillary_vertical = MD3Motor("CapillaryVertical", MD3_CLIENT)
    capillary_horizontal = MD3Motor("CapillaryHorizontal", MD3_CLIENT)
    scintillator_vertical = MD3Motor("ScintillatorVertical", MD3_CLIENT)
    beamstop_x = MD3Motor("BeamstopX", MD3_CLIENT)
    beamstop_y = MD3Motor("BeamstopY", MD3_CLIENT)
    beamstop_z = MD3Motor("BeamstopZ", MD3_CLIENT)
    zoom = MD3Zoom("Zoom", MD3_CLIENT)
    phase = MD3Phase("Phase", MD3_CLIENT)
    backlight = MD3BackLight("Backlight", MD3_CLIENT)
    frontlight = MD3FrontLight("Frontlight", MD3_CLIENT)
    plate_translation = MD3PLateTranslation("PlateTranslation", MD3_CLIENT)
    move_plate_to_shelf = MD3MovePlateToShelf("MovePlateToShelf", MD3_CLIENT)
    beam_center = BeamCenter("beam_center", MD3_CLIENT)
    focus = MD3Focus("CentringTableFocusPosition", MD3_CLIENT)
    fast_shutter = MD3FastShutter("FastShutter", MD3_CLIENT)

    @property
    def state(self) -> str:
        """Gets the state of the md3

        Returns
        -------
        str
            The state of the md3
        """
        return MD3_CLIENT.getState()

    def restart(self, cold_restart: bool = False) -> str:
        """
        Restarts the md3

        Parameters
        ----------
        cold_restart : bool, optional
            If true, we execute a cold restart, by default False

        Returns
        -------
        str
            Usually returns `null` if the restart is successful
        """
        return MD3_CLIENT.restart(cold_restart)

    def abort(self) -> str:
        """
        Aborts an MD3 action

        Returns
        -------
        str
            Usually returns `null` if the restart is successful
        """
        return MD3_CLIENT.abort()

    def set_beam_center(self, beam_center: tuple[int, int], zoom_level: int):
        self.zoom.set(zoom_level)
        sleep(0.1)

        MD3_CLIENT.setBeamPositionHorizontal(beam_center[0])
        MD3_CLIENT.setBeamPositionVertical(beam_center[1])

    def get_beam_center(self) -> tuple[int, int]:
        x = MD3_CLIENT.getBeamPositionHorizontal()
        y = MD3_CLIENT.getBeamPositionVertical()
        return (x, y)

    def save_centring_position(self) -> None:
        MD3_CLIENT.saveCentringPositions()

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
        return MD3_CLIENT.getHeadType()
