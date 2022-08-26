""" Motor Definitions """

from ophyd import Component as Cpt, EpicsMotor, MotorBundle
from epics.pv import fmt_time

from ophyd.signal import (EpicsSignal, EpicsSignalRO)
from ophyd.utils import DisconnectedError
from ophyd.utils.epics_pvs import raise_if_disconnected
from ophyd.positioner import PositionerBase
from ophyd.device import (Device, Component as Cpt)
from ophyd.status import wait as status_wait
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Testrig(MotorBundle):
    """
    Testrig motors
    """

    x = Cpt(EpicsMotor, ":mtr2", lazy=True)
    y = Cpt(EpicsMotor, ":mtr3", lazy=True)
    z = Cpt(EpicsMotor, ":mtr1", lazy=True)


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


class HomeEnum(str, Enum):
    forward = "forward"
    reverse = "reverse"


class CosylabMotor(Device, PositionerBase):
    '''An EPICS motor record, wrapped in a :class:`Positioner`

    Keyword arguments are passed through to the base class, Positioner

    Parameters
    ----------
    prefix : str
        The record to use
    read_attrs : sequence of attribute names
        The signals to be read during data acquisition (i.e., in read() and
        describe() calls)
    name : str, optional
        The name of the device
    parent : instance or None
        The instance of the parent device, if applicable
    settle_time : float, optional
        The amount of time to wait after moves to report status completion
    timeout : float, optional
        The default timeout to use for motion requests, in seconds.
    '''
    user_offset = Cpt(EpicsSignal, '_OFFSET_SP')
    user_readback = Cpt(EpicsSignalRO, '_MON')
    user_setpoint = Cpt(EpicsSignal, '_SP', limits=True)
    motor_egu = Cpt(EpicsSignal, '_SP.EGU')
    motor_is_moving = Cpt(EpicsSignalRO, '_IN_POSITION_STS')
    motor_done_move = Cpt(EpicsSignalRO, '_INPOS_STS')
    motor_stop = Cpt(EpicsSignal, '_STOP_MOTION_CMD.PROC')
    # TCD; move mode. 0=PAUSE, 1=GO
    move_mode = Cpt(EpicsSignal, '_MV_MODE_CMD')
    # TCD; trigger move (if paused)
    trigger_move = Cpt(EpicsSignal, '_MV_CMD.PROC')
    #offset_freeze_switch = Cpt(EpicsSignal, '.FOFF')
    velocity = Cpt(EpicsSignal, '_ACVES_MON')
    acceleration = Cpt(EpicsSignal, '_RAW_MAX_ACC_MON')
    #set_use_switch = Cpt(EpicsSignal, '.SET')
    high_limit_switch = Cpt(EpicsSignal, '_HIGH_LIMIT_STS')
    low_limit_switch = Cpt(EpicsSignal, '_LOW_LIMIT_STS')
    home_forward = Cpt(EpicsSignal, '_RAW_HOME_CMD.PROC')
    #home_reverse = Cpt(EpicsSignal, '.HOMR')
    direction_of_travel = Cpt(EpicsSignal, '_DIRECTION')

    def __init__(self, prefix, *, read_attrs=None, configuration_attrs=None,
                 name=None, parent=None, **kwargs):
        if read_attrs is None:
            read_attrs = ['user_readback', 'user_setpoint']

        if configuration_attrs is None:
            configuration_attrs = ['motor_egu', ]

        super().__init__(prefix, read_attrs=read_attrs,
                         configuration_attrs=configuration_attrs,
                         name=name, parent=parent, **kwargs)

        # Make the default alias for the user_readback the name of the
        # motor itself.
        self.user_readback.name = self.name

        self.motor_done_move.subscribe(self._move_changed)
        self.user_readback.subscribe(self._pos_changed)

    @property
    @raise_if_disconnected
    def precision(self):
        '''The precision of the readback PV, as reported by EPICS'''
        return self.user_readback.precision

    @property
    @raise_if_disconnected
    def egu(self):
        '''The engineering units (EGU) for a position'''
        return self.motor_egu.get()

    @property
    @raise_if_disconnected
    def limits(self):
        return self.user_setpoint.limits

    @property
    @raise_if_disconnected
    def moving(self):
        '''Whether or not the motor is moving

        Returns
        -------
        moving : bool
        '''
        # TCD; invert bool as 1 is in position now
        return not bool(self.motor_is_moving.get(use_monitor=False))

    @raise_if_disconnected
    def stop(self):
        self.motor_stop.put(1, wait=False)
        super().stop()

    @raise_if_disconnected
    def move(self, position, wait=True, **kwargs):
        '''Move to a specified position, optionally waiting for motion to
        complete.

        Parameters
        ----------
        position
            Position to move to
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

        Raises
        ------
        TimeoutError
            When motion takes longer than `timeout`
        ValueError
            On invalid positions
        RuntimeError
            If motion fails other than timing out
        '''

        
        self._started_moving = False

        status = super().move(position, **kwargs)
        # TCD; add go/pause tests
        if self.move_mode == 1:
            self.user_setpoint.put(position, wait=False)
        else:
            self.user_setpoint.put(position, wait=False)
            self.trigger_move.put(1, wait=False)

        # self.user_setpoint.put(position, wait=False)

        try:
            if wait:
                status_wait(status)
        except KeyboardInterrupt:
            self.stop()
            raise
        

        """
        self._started_moving = False

        status = super().move(position, **kwargs)
        self.user_setpoint.put(position, wait=False)
        try:
            if wait:
                status_wait(status)
        except KeyboardInterrupt:
            self.stop()
            raise
            """

        return status

    @property
    @raise_if_disconnected
    def position(self):
        '''The current position of the motor in its engineering units

        Returns
        -------
        position : float
        '''
        return self._position

    @raise_if_disconnected
    def set_current_position(self, pos):
        '''Configure the motor user position to the given value

        Parameters
        ----------
        pos
           Position to set.

        '''
        # TCD; disable epics .SET as not an option
        #self.set_use_switch.put(1, wait=True)
        self.user_setpoint.put(pos, wait=True)
        #self.set_use_switch.put(0, wait=True)

    @raise_if_disconnected
    def home(self, direction, wait=True, **kwargs):
        '''Perform the default homing function in the desired direction

        Parameters
        ----------
        direction : HomeEnum
           Direction in which to perform the home search.
        '''
        # TCD;Only use forward homing as nto an option with pmac
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

    def check_value(self, pos):
        '''Check that the position is within the soft limits'''
        self.user_setpoint.check_value(pos)

    def _pos_changed(self, timestamp=None, value=None, **kwargs):
        '''Callback from EPICS, indicating a change in position'''
        self._set_position(value)

    def _move_changed(self, timestamp=None, value=None, sub_type=None,
                      **kwargs):
        '''Callback from EPICS, indicating that movement status has changed'''
        was_moving = self._moving
        # TCD; may need to be inverted TBA
        self._moving = (value != 1)

        started = False
        if not self._started_moving:
            started = self._started_moving = (not was_moving and self._moving)

        logger.debug('[ts=%s] %s moving: %s (value=%s)', fmt_time(timestamp),
                     self, self._moving, value)

        if started:
            self._run_subs(sub_type=self.SUB_START, timestamp=timestamp,
                           value=value, **kwargs)

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
    def report(self):
        try:
            rep = super().report
        except DisconnectedError:
            # TODO there might be more in this that gets lost
            rep = {'position': 'disconnected'}
        rep['pv'] = self.user_readback.pvname
        return rep
