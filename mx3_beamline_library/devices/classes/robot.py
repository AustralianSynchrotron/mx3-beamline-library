from os import environ
from time import sleep

from mx_robot_library.client import Client
from mx_robot_library.schemas.common.path import RobotPaths
from mx_robot_library.schemas.responses.state import StateResponse
from ophyd import Component as Cpt, Device, Signal, SignalRO

ROBOT_HOST = environ.get("ROBOT_HOST", "12.345.678.9")
# Create a new client instance
CLIENT = Client(
    host=ROBOT_HOST,  # Controller IP
    status_port=1000,  # Status query port [ENV: MX_ASC_STATUS_PORT"]
    cmd_port=10000,  # Trajectory command port [ENV: ASC_CMD_PORT"]
    readonly=False,  # Toggle to block trajectory calls (Local to client)
)


class State(SignalRO):
    """
    Ophyd device used to read the state of the robot
    """

    def __init__(self, name: str, client: Client, *args, **kwargs) -> None:
        """
        Parameters
        ----------
        name : str
            Signal name
        client : Client
            A robot client object

        Returns
        -------
        None
        """
        super().__init__(name=name, *args, **kwargs)

        self.client = client
        self.name = name

    def get(self) -> StateResponse:
        """
        Gets state of the robot

        Returns
        -------
        StateResponse
            The current phase
        """
        return self.state

    @property
    def state(self) -> StateResponse:
        """
        Gets state of the robot

        Returns
        -------
        StateResponse
            The current phase
        """
        return self.client.status.state


class Mount(Signal):
    """
    Ophyd device used to mount a pin
    """

    def __init__(self, name: str, client: Client, *args, **kwargs) -> None:
        """
        Parameters
        ----------
        name : str
            Signal name
        client : Client
            A robot client object

        Returns
        -------
        None
        """
        super().__init__(name=name, *args, **kwargs)

        self.client = client
        self.name = name

    def get(self) -> str:
        """
        Checks if there is a pin on the goniometer

        Returns
        -------
        StateResponse
            State response
        """
        return self.client.status.state.goni_pin

    def _set_and_wait(self, value: dict, timeout: float = None) -> bytes:
        """
        Sends the mount command to the robot.

        Parameters
        ----------
        value : float
            The value
        timeout : float, optional
            Maximum time to wait for value to be successfully set, or None

        Returns
        -------
        bytes
            The robot response
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
    Ophyd device used to unmount a pin
    """

    def __init__(self, name: str, client: Client, *args, **kwargs) -> None:
        """
        Parameters
        ----------
        name : str
            Signal name
        client : Client
            A robot client object

        Returns
        -------
        None
        """
        super().__init__(name=name, *args, **kwargs)

        self.client = client
        self.name = name

    def get(self) -> StateResponse:
        """
        Checks if there is a pin on the goniometer

        Returns
        -------
        StateResponse
            State response
        """
        return self.client.status.state.goni_pin

    def _set_and_wait(self, value=None, timeout: float = None) -> bytes:
        """
        Sends the unmount command to the robot.

        Parameters
        ----------
        value : float
            The value
        timeout : float, optional
            Maximum time to wait for value to be successfully set, or None

        Returns
        -------
        bytes
            The robot response
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


class IsaraRobot(Device):
    state = Cpt(State, name="state", client=CLIENT)
    mount = Cpt(Mount, name="mount", client=CLIENT)
    unmount = Cpt(Unmount, name="unmount", client=CLIENT)
