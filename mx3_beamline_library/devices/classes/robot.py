from mx_robot_library.client import Client
from ophyd import Signal, SignalRO
from ophyd import Device, Component as Cpt
from time import sleep
from mx_robot_library.schemas.common.path import RobotPaths
from os import environ


class State(SignalRO):
    """
    Ophyd device used to control the phase of the MD3.
    The accepted phases are Centring, DataCollection, BeamLocation, and
    Transfer
    """

    def __init__(self, name: str, client: Client, *args, **kwargs) -> None:
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

        self.client = client
        self.name = name

    def get(self) -> str:
        """Gets the current phase

        Returns
        -------
        str
            The current phase
        """
        return self.state

    @property
    def state(self):
        return self.client.status.state
    
class Mount(Signal):
    """
    Ophyd device used to control the phase of the MD3.
    The accepted phases are Centring, DataCollection, BeamLocation, and
    Transfer
    """

    def __init__(self, name: str, client: Client, *args, **kwargs) -> None:
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

        self.client = client
        self.name = name

    def get(self) -> str:
        """Gets the current phase

        Returns
        -------
        str
            The current phase
        """
        return self.client.status.state.path


    def _set_and_wait(self, value: dict, timeout: float = None) -> None:
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
        
        pin = self.client.utils.get_pin(value["id"], value["puck"])
        msg = self.client.trajectory.mount(pin=pin)

        # Wait until operation is complete
        sleep(0.5)
        while self.client.status.state.path != RobotPaths.UNDEFINED:
            sleep(0.5)
        assert self.client.status.state.goni_pin == pin

        return msg
    

class Unmount(Signal):
    """
    Ophyd device used to control the phase of the MD3.
    The accepted phases are Centring, DataCollection, BeamLocation, and
    Transfer
    """

    def __init__(self, name: str, client: Client, *args, **kwargs) -> None:
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

        self.client = client
        self.name = name

    def get(self) -> str:
        """Gets the current phase

        Returns
        -------
        str
            The current phase
        """
        return self.client.status.state.path


    def _set_and_wait(self, value = None, timeout: float = None) -> None:
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
        
        # Try to unmount a pin
        msg = self.client.trajectory.unmount()

        # Wait until operation is complete
        sleep(0.5)
        while self.client.status.state.path != RobotPaths.UNDEFINED:
            sleep(0.5)
        # Check there is no pin on the goniometer
        assert self.client.status.state.goni_pin is None

        return msg


ROBOT_HOST = environ.get("ROBOT_HOST", "12.345.678.9")
# Create a new client instance
CLIENT = Client(
    host=ROBOT_HOST,  # Controller IP
    status_port=1000,  # Status query port [ENV: MX_ASC_STATUS_PORT"]
    cmd_port=10000,  # Trajectory command port [ENV: ASC_CMD_PORT"]
    readonly=False,  # Toggle to block trajectory calls (Local to client)
)


class IsaraRobot(Device):
    state = Cpt(State, name="state", client=CLIENT)
    mount = Cpt(Mount, name="mount", client=CLIENT)
    unmount = Cpt(Unmount, name="unmount", client=CLIENT)


if __name__ == "__main__":
    from bluesky import RunEngine
    from bluesky.callbacks.best_effort import BestEffortCallback
    from bluesky.plan_stubs import mv

    
    isara_robot = IsaraRobot(name="robot")

    print(isara_robot.state.get())

    RE = RunEngine()
    bec = BestEffortCallback()
    RE.subscribe(bec)

    def mount_sample(mount_signal: Mount, id: int, puck: int):
        """
        Mounts a sample given an id and puck

        Parameters
        ----------
        mount_signal : Mount
            A robot mount signal
        id : int
            id
        puck : int
            Puck

        Yields
        ------
        _type_
            _description_
        """
        yield from mv(mount_signal, {"id": id ,"puck": puck})

    def unmount_sample(unmount_signal: Unmount):
        yield from mv(unmount_signal, None)

    mount_signal = isara_robot.mount
    unmount_signal = isara_robot.unmount

    #RE(mount_sample(mount_signal, id=1, puck=1))
    RE(unmount_sample(unmount_signal))

