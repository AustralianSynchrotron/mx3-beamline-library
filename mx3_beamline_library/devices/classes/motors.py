""" Motor Definitions """

# from as_acquisition_library.devices.motors import CosylabMotor
import logging
from os import environ, path
from time import sleep

import numpy as np
import yaml
from ophyd import Component as Cpt, EpicsMotor, MotorBundle, Signal
from ophyd.device import Device, required_for_connection
from ophyd.epics_motor import HomeEnum
from ophyd.positioner import PositionerBase
from ophyd.signal import EpicsSignal, EpicsSignalRO
from ophyd.status import MoveStatus, Status, wait as status_wait
from ophyd.utils import DisconnectedError
from ophyd.utils.epics_pvs import AlarmSeverity, raise_if_disconnected

from .md3.ClientFactory import ClientFactory

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)


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

    def _set_and_wait(self, value: float, timeout: float = 20) -> None:
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
        initial_position = self.get()

        if timeout is None:
            logger.info("Cannot pass timeout=None to the server. Setting timeout=20")
            timeout = 20

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

        path_to_config_file = path.join(
            path.dirname(__file__),
            "../../plans/configuration/optical_and_xray_centering.yml",
        )
        with open(path_to_config_file, "r") as plan_config:
            plan_args: dict = yaml.safe_load(plan_config)

        self._pixels_per_mm = plan_args["md3_camera"]["pixels_per_mm"]

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
        try:
            logger.info(f"Changing MD3 phase from {self.get()} to: {value}")
            # Check if there is an activity still running and wait
            # until this activity has finished before executing another command
            current_phase = self.get()
            while current_phase == "Unknown":
                sleep(0.2)
                current_phase = self.get()

            self.server.startSetPhase(value)

            # There is not a wait function on the MD3 phase setter, so the following
            # block a waits for the MD3 to change phase
            current_phase = self.get()
            while current_phase == "Unknown":
                sleep(0.2)
                current_phase = self.get()
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

    def get(self) -> str:
        """Gets the current phase

        Returns
        -------
        str
            The current phase
        """
        return self.server.getBackLightFactor()

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
            self.server.setBackLightFactor(value)
        else:
            logger.info(f"Allowed values are: {self.allowed_values}, not {value}")


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
            self.server.setFrontLightFactor(value)
        else:
            logger.info(f"Allowed values are: {self.allowed_values}, not {value}")


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

        self.server.setPlateTranslationPosition(value)


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

    def get(self) -> tuple[int, int, int]:
        """Gets the current drop location

        Returns
        -------
        tuple[int, int, int]
            The current row, column, and drop location
        """
        return self.server.getDropLocation()

    def _set_and_wait(self, value: tuple[int, int, int], timeout: float = None) -> None:
        """
        Moves the plate to the specified drop position (row, columns, drop)

        Parameters
        ----------
        value : tuple[int, int, int]
            The drop locations
        timeout : float, optional
            Maximum time to wait for value to be successfully set, or None

        Returns
        -------
        None
        """

        self.server.movePlateToShelf(value[0], value[1], value[2])


MD3_ADDRESS = environ.get("MD3_ADDRESS", "10.244.101.30")
MD3_PORT = int(environ.get("MD3_PORT", 9001))

SERVER = ClientFactory.instantiate(
    type="exporter", args={"address": MD3_ADDRESS, "port": MD3_PORT}
)


class MicroDiffractometer:
    """
    MD3 motors
    """

    sample_x = MD3Motor("CentringX", SERVER)
    sample_y = MD3Motor("CentringY", SERVER)
    alignment_x = MD3Motor("AlignmentX", SERVER)
    alignment_y = MD3Motor("AlignmentY", SERVER)
    alignment_z = MD3Motor("AlignmentZ", SERVER)
    omega = MD3Motor("Omega", SERVER)
    kappa = MD3Motor("Kappa", SERVER)
    phi = MD3Motor("Phi", SERVER)  # This motor is named Kappa phi in mxcube
    aperture_vertical = MD3Motor("ApertureVertical", SERVER)
    aperture_horizontal = MD3Motor("ApertureHorizontal", SERVER)
    capillary_vertical = MD3Motor("CapillaryVertical", SERVER)
    capillary_horizontal = MD3Motor("CapillaryHorizontal", SERVER)
    scintillator_vertical = MD3Motor("ScintillatorVertical", SERVER)
    beamstop_x = MD3Motor("BeamstopX", SERVER)
    beamstop_y = MD3Motor("BeamstopY", SERVER)
    beamstop_z = MD3Motor("BeamstopZ", SERVER)
    zoom = MD3Zoom("Zoom", SERVER)
    phase = MD3Phase("Phase", SERVER)
    backlight = MD3BackLight("Backlight", SERVER)
    frontlight = MD3FrontLight("Frontlight", SERVER)
    plate_translation = MD3PLateTranslation("PlateTranslation", SERVER)
    move_plate_to_shelf = MD3MovePlateToShelf("MovePlateToShelf", SERVER)
