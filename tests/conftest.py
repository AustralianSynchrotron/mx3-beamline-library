import os
import typing
import pytest

if typing.TYPE_CHECKING:
    from _pytest.fixtures import SubRequest
    from mx3_beamline_library import devices
    from mx3_beamline_library.devices.sim import detectors, motors
    from mx3_beamline_library.devices.classes.detectors import DectrisDetector
    Devices = devices
    Motors = motors
    Detectors = detectors


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
