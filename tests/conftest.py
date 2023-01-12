import os
import typing
import pytest
from bluesky import RunEngine

if typing.TYPE_CHECKING:
    from _pytest.fixtures import SubRequest
    from ophyd import MotorBundle
    from mx3_beamline_library import devices
    from mx3_beamline_library.devices.sim import detectors, motors
    from mx3_beamline_library.devices.classes.detectors import DectrisDetector
    from mx3_beamline_library.devices.sim.classes.detectors import SimBlackFlyCam
    from mx3_beamline_library.devices.sim.classes.motors import MX3SimMotor
    Devices = devices
    Motors = motors
    Detectors = detectors
    Cameras = Detectors


@pytest.fixture(scope="session")
def devices() -> "Devices":
    """Pytest fixture to return devices module.

    Returns
    -------
    Devices
        Simulated devices module.
    """

    # Import beamline library with sim devices enabled
    os.environ["BL_ACTIVE"] = "false"
    from mx3_beamline_library import devices
    from mx3_beamline_library.devices import motors, detectors

    # Check beamline library was initialised with sim devices enabled
    from mx3_beamline_library.devices.sim import motors as sim_motors, detectors as sim_detectors
    assert motors == sim_motors
    assert detectors == sim_detectors

    return devices


@pytest.fixture(scope="session")
def motors(devices: "Devices") -> "Motors":
    """Pytest fixture to return motors module.

    Parameters
    ----------
    devices : Devices
        Simulated devices module.

    Returns
    -------
    Motors
        Simulated motors module.
    """

    from mx3_beamline_library.devices import motors
    return motors


@pytest.fixture(scope="session")
def detectors(devices: "Devices") -> "Detectors":
    """_summary_

    Parameters
    ----------
    devices : Devices
        Simulated devices module.

    Returns
    -------
    Detectors
        Simulated detectors module.
    """

    from mx3_beamline_library.devices import detectors
    return detectors


@pytest.fixture(scope="class")
def detector(request: "SubRequest", detectors: "Detectors") -> "DectrisDetector":
    """Pytest fixture to load detector Ophyd device.

    Parameters
    ----------
    request : SubRequest
        Pytest subrequest parameters.
    detectors : Detectors
        Loaded detector module, either simulated or real.

    Returns
    -------
    DectrisDetector
        Dectris detector device instance.
    """

    # Load Ophyd device
    detector_name = request.param
    detector: "DectrisDetector" = getattr(detectors, detector_name)
    detector.wait_for_connection(timeout=5)
    yield detector


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


@pytest.fixture(scope="class")
def camera(request: "SubRequest", detectors: "Detectors") -> "SimBlackFlyCam":
    """Pytest fixture to load camera Ophyd device.

    Parameters
    ----------
    request : SubRequest
        Pytest subrequest parameters.
    detectors : Detectors
        Loaded detector module, either simulated or real.

    Returns
    -------
    SimBlackFlyCam
        Camera device instance.
    """

    # Load Ophyd device
    camera_name = request.param
    camera: "SimBlackFlyCam" = getattr(detectors, camera_name)
    camera.wait_for_connection(timeout=5)
    yield camera


@pytest.fixture(scope="class")
def run_engine() -> RunEngine:
    """Pytest fixture to initialise the bluesky run engine.

    Returns
    -------
    RunEngine
        Instance of the bluesky run engine.
    """

    yield RunEngine({})
