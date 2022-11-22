from time import sleep
import typing
import pytest
from typing import Union
from ophyd.utils import LimitError

if typing.TYPE_CHECKING:
    from _pytest.fixtures import SubRequest
    from ophyd import MotorBundle
    from mx3_beamline_library.devices.sim.classes.motors import MX3SimMotor
    from mx3_beamline_library.devices.sim import motors
    Motors = motors


@pytest.fixture(scope="class")
def motor(request: "SubRequest", motors: "Motors") -> "MX3SimMotor":
    """Pytest fixture to load motor Ophyd devices.

    Parameters
    ----------
    request : SubRequest
        Pytest subrequest parameters.
    motors : Motors
        Loaded motors module, either simulated or real.

    Returns
    -------
    MX3SimMotor
        Motor device instance.
    """

    device_name, motor_name = request.param
    device: "MotorBundle" = getattr(motors, device_name)
    motor: "MX3SimMotor" = getattr(device, motor_name)
    motor.wait_for_connection(timeout=300)

    return motor


@pytest.mark.parametrize(
    "motor",
    [
        ("testrig", "x"),
        ("testrig", "y"),
        ("testrig", "z"),
        ("mxcube_sim_PVs", "m1"),
        ("mxcube_sim_PVs", "m2"),
        ("mxcube_sim_PVs", "m3"),
        ("mxcube_sim_PVs", "m4"),
        ("mxcube_sim_PVs", "m5"),
        ("mxcube_sim_PVs", "m6"),
        ("mxcube_sim_PVs", "m7"),
        ("mxcube_sim_PVs", "m8"),
    ],
    indirect=True,
)
class TestMotors:
    """Run motor tests."""

    def test_motor_setup(self, motor: "MX3SimMotor"):
        """Test motor device initialised.

        Parameters
        ----------
        motor : MX3SimMotor
            Motor device instance.
        """

        assert motor is not None

    def test_get_limits(self, motor: "MX3SimMotor"):
        """Test getting motor limits.

        Parameters
        ----------
        motor : MX3SimMotor
            Motor device instance.
        """

        # Check return value typing
        limits = motor.limits
        assert isinstance(limits, tuple)
        assert len(limits) == 2
        assert isinstance(limits[0], float) and isinstance(limits[1], float)

        # Check upper and lower limits
        assert limits[0] < 0
        assert limits[1] > 0

    def test_get_velocity(self, motor: "MX3SimMotor"):
        """Test getting motor velocity.

        Parameters
        ----------
        motor : MX3SimMotor
            Motor device instance.
        """

        velocity = motor.velocity.get()

        # Assuming int/float for typing here, to be confirmed
        assert isinstance(velocity, int) or isinstance(velocity, float)

    @pytest.mark.parametrize("value", [3, 4, 5.6, 9.99, 2.0])
    def test_put_velocity(self, value: Union[int, float], motor: "MX3SimMotor"):
        """Test setting motor velocity.

        Parameters
        ----------
        motor : MX3SimMotor
            Motor device instance.
        """

        # Put velocity
        motor.velocity.put(value=value)

        # Validate velocity has been updated
        assert motor.velocity.get() == value

    def test_get_position(self, motor: "MX3SimMotor"):
        """Test getting motor position.

        Parameters
        ----------
        motor : MX3SimMotor
            Motor device instance.
        """

        position = motor.position

        # Assuming int/float for typing here, to be confirmed
        assert isinstance(position, int) or isinstance(position, float)

    @pytest.mark.parametrize("value", [-10, -2.5, 0, 1, 2.5, 10])
    def test_move_motor(self, value: Union[int, float], motor: "MX3SimMotor"):
        """Test moving motor.

        Parameters
        ----------
        motor : MX3SimMotor
            Motor device instance.
        """

        motor.move(value, wait=True)

    def test_move_limits(self, motor: "MX3SimMotor"):
        """Test moving motor outside limits.

        Parameters
        ----------
        motor : MX3SimMotor
            Motor device instance.
        """

        lower_limit, upper_limit = motor.limits

        # Test lower limit
        with pytest.raises(LimitError):
            motor.move(lower_limit - 1.0, wait=True)

        # Test upper limit
        with pytest.raises(LimitError):
            motor.move(upper_limit + 1.0, wait=True)

    def test_moving(self, motor: "MX3SimMotor"):
        """Test motor moving flag.

        Parameters
        ----------
        motor : MX3SimMotor
            Motor device instance.
        """

        # Move motor
        motor.move(0.0, wait=False)

        # Check sim motor moving
        assert motor.moving

        sleep(motor.delay)

        # Check sim motor has stoped moving
        assert not motor.moving
