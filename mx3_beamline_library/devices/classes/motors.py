""" Motor Definitions """

from as_acquisition_library.devices.motors import CosylabMotor
from ophyd import Component as Cpt, EpicsMotor, MotorBundle
from ophyd.status import MoveStatus, wait as status_wait
from ophyd.utils.epics_pvs import raise_if_disconnected


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


class MX3CosylabMotor(CosylabMotor):
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


class Testrig(MotorBundle):
    """
    Testrig motors
    """

    x = Cpt(MX3CosylabMotor, ":X", lazy=True)
    y = Cpt(MX3CosylabMotor, ":Y", lazy=True)
    z = Cpt(MX3CosylabMotor, ":Z", lazy=True)
    phi = Cpt(MX3CosylabMotor, ":PHI", lazy=True)
