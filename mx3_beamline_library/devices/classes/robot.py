from os import environ
from time import sleep

from mx_robot_library.client import Client
from mx_robot_library.schemas.common.path import RobotPaths
from mx_robot_library.schemas.common.sample import Plate
from mx_robot_library.schemas.responses.state import StateResponse
from ophyd import Component as Cpt, Device, Signal, SignalRO

from ...logger import setup_logger
from .motors import MD3_CLIENT

logger = setup_logger(__name__)


ROBOT_HOST = environ.get("ROBOT_HOST", "127.0.0.1")
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

    def _set_and_wait(self, value: dict, timeout: float = None) -> None:
        """
        Sends the mount command to the robot.

        Parameters
        ----------
        value : dict
            A dictionary containing the id and puck
        timeout : float, optional
            Maximum time to wait for value to be successfully set, or None

        Returns
        -------
        None
        """
        try:
            pin = value["pin"]
            prepick_pin = value["prepick_pin"]
            # Mount pin on goni
            if self.client.status.state.goni_pin is not None:
                self.client.trajectory.puck.unmount_then_mount(
                    pin=pin, prepick_pin=prepick_pin, wait=True
                )
            else:
                self.client.trajectory.puck.mount(
                    pin=pin, prepick_pin=prepick_pin, wait=True
                )

            # Wait until operation is complete
            sleep(0.3)
            while self.client.status.state.path != RobotPaths.UNDEFINED:
                sleep(0.3)
            assert (
                self.client.status.state.goni_pin == pin
            ), f"Unable to mount pin {pin}"

            while MD3_CLIENT.getState() != "Ready":
                sleep(0.5)
        except Exception as ex:
            # This will also display the cause of the exception when using the
            # bluesky run engine
            self._status.set_exception(ex)


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

    def _set_and_wait(self, value=None, timeout: float = None) -> None:
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
        None
        """
        try:
            # Check if the robot has pre-picked a pin
            if self.client.status.state.jaw_a_pin is not None:
                # TODO: test return_pin(wait=True)
                self.client.trajectory.puck.return_pin()
                # Wait until operation is complete
                sleep(0.5)
                while self.client.status.state.path != RobotPaths.UNDEFINED:
                    sleep(0.5)

            # Try to unmount a pin
            self.client.trajectory.puck.unmount(wait=True)

            # Wait until operation is complete
            sleep(0.5)
            while self.client.status.state.path != RobotPaths.UNDEFINED:
                sleep(0.5)
            # Check there is no pin on the goniometer
            assert (
                self.client.status.state.goni_pin is None
            ), "The robot has probably failed unmounting the pin"

            while MD3_CLIENT.getState() != "Ready":
                sleep(0.5)
        except Exception as ex:
            # This will also display the cause of the exception when using the
            # bluesky run engine
            self._status.set_exception(ex)


class MountTray(Signal):
    """
    Ophyd signal used to mount a tray
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
        Checks if there is a tray on the goniometer

        Returns
        -------
        StateResponse
            State response
        """
        return self.client.status.state.goni_plate

    def _set_and_wait(self, value: int, timeout: float = None) -> None:
        """
        Sends the mount command to the robot.

        Parameters
        ----------
        value : dict
            A dictionary containing the id of the tray
        timeout : float, optional
            Maximum time to wait for value to be successfully set, or None

        Returns
        -------
        None
        """
        try:
            PLATE_TO_MOUNT = Plate(id=value)

            # Mount plate from position "1"
            self.client.trajectory.plate.mount(plate=PLATE_TO_MOUNT, wait=True)

            assert (
                self.client.status.state.goni_plate == PLATE_TO_MOUNT
            ), "Mount unsuccessful"

            while MD3_CLIENT.getState() != "Ready":
                sleep(0.5)
        except Exception as ex:
            # This will also display the cause of the exception when using the
            # bluesky run engine
            self._status.set_exception(ex)


class UnmountTray(Signal):
    """
    Ophyd signal used to unmount a tray
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
        Checks if there is a tray on the goniometer

        Returns
        -------
        StateResponse
            State response
        """
        return self.client.status.state.goni_plate

    def _set_and_wait(self, value=None, timeout: float = None) -> None:
        """
        Sends the mount command to the robot.

        Parameters
        ----------
        value : dict
            A dictionary containing the id and puck
        timeout : float, optional
            Maximum time to wait for value to be successfully set, or None

        Returns
        -------
        None
        """
        try:
            # Unmount plate from goni
            self.client.trajectory.plate.unmount(wait=True)

            assert self.client.status.state.goni_plate is None, "Unmount unsuccessful"

            while MD3_CLIENT.getState() != "Ready":
                sleep(0.5)
        except Exception as ex:
            # This will also display the cause of the exception when using the
            # bluesky run engine
            self._status.set_exception(ex)


class IsaraRobot(Device):
    state = Cpt(State, name="state", client=CLIENT)
    mount = Cpt(Mount, name="mount", client=CLIENT)
    unmount = Cpt(Unmount, name="unmount", client=CLIENT)
    mount_tray = Cpt(MountTray, name="mount_tray", client=CLIENT)
    unmount_tray = Cpt(UnmountTray, name="unmount_tray", client=CLIENT)
